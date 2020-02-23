#!/usr/bin/env python3
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
help_fd = "The ratio of focal length to sensor height (default == 905/590)"
help_dh = "Dot's height (default == 1)"
ap.add_argument("-fd", "--fdratio", required=False, help=help_fd)
ap.add_argument("-dh", "--dotheight", required=False, help=help_dh)
args = vars(ap.parse_args())

if args["fdratio"] is not None:
    fd_ratio = float(args["fdratio"])
else:
    fd_ratio = 905/590

if args["dotheight"] is not None:
    dot_height = float(args["dotheight"])
else:
    dot_height = 20
cap = cv2.VideoCapture(0) #default
#cap = cv2.VideoCapture('http://192.168.42.129:8080/video/mjpeg')    
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

multiplier = fd_ratio * dot_height * height / 20

# Uncomment to output the recording
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("output.avi", fourcc, 20.0, (width, height), True)

mask = np.zeros((height, width), dtype="uint8")
#size = 200 #default ukuran kotak
size = 50
cv2.rectangle(mask, (width//2 - size//2, height//2 - size//2),
              (width//2 + size//2, height//2 + size//2), (255), -1)
while cap.isOpened():
    ret, image = cap.read()
    if ret == 0:
        break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 30, 150)
    (cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    cv2.rectangle(image, (width//2 - size//2, height//2 - size//2),
                  #(width//2 + size//2, height//2 + size//2), (255, 0, 0), 1)
                  (width//2 + size//2, height//2 + size//2), (255, 0, 0), 1)

    for (i, c) in enumerate(cnts):
        ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
        if centerX >= width//2 - size//2 and centerX <= width//2 + size//2:
            if centerY >= height//2 - size//2 and \
               centerY <= height//2 + size//2:
                if radius <= size//2:
                    #distance = multiplier/radius
                    distance = multiplier/radius
                    cv2.circle(image, (int(centerX), int(centerY)),
                               int(radius), (0, 255, 0), -1)
                    cv2.putText(image, "{:.2f} cm".format(distance),
                                (int(centerX), int(centerY)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
    cv2.drawContours(image, cnts, -1, (0, 0, 255), 1)

    # Uncomment to output the recording
    # out.write(image)
    cv2.imshow("out", image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

cap.release()

# Uncomment to output the recording
# out.release()
cv2.destroyAllWindows()
