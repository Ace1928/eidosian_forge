import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def setXRotation(self, angle):
    angle = self.normalizeAngle(angle)
    if angle != self.xRot:
        self.xRot = angle
        self.emit(QtCore.SIGNAL('xRotationChanged(int)'), angle)
        self.update()