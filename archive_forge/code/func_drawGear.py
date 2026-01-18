import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
def drawGear(self, gear, dx, dy, dz, angle):
    glPushMatrix()
    glTranslated(dx, dy, dz)
    glRotated(angle, 0.0, 0.0, 1.0)
    glCallList(gear)
    glPopMatrix()