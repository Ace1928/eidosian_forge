import sys
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
import textures_rc
def rotateBy(self, xAngle, yAngle, zAngle):
    self.xRot = (self.xRot + xAngle) % 5760
    self.yRot = (self.yRot + yAngle) % 5760
    self.zRot = (self.zRot + zAngle) % 5760
    self.updateGL()