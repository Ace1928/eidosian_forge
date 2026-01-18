import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def extrude(self, x1, y1, x2, y2):
    darkTrolltechGreen = self.trolltechGreen.darker(250 + int(100 * x1))
    GL.glColor(darkTrolltechGreen.redF(), darkTrolltechGreen.greenF(), darkTrolltechGreen.blueF(), darkTrolltechGreen.alphaF())
    GL.glVertex3d(x1, y1, -0.05)
    GL.glVertex3d(x2, y2, -0.05)
    GL.glVertex3d(x2, y2, +0.05)
    GL.glVertex3d(x1, y1, +0.05)