import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def resizeGL(self, width, height):
    side = min(width, height)
    GL.glViewport(int((width - side) / 2), int((height - side) / 2), side, side)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GL.glOrtho(-0.5, +0.5, -0.5, +0.5, 4.0, 15.0)
    GL.glMatrixMode(GL.GL_MODELVIEW)