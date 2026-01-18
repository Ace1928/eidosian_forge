import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def setZRotation(self, angle):
    angle = self.normalizeAngle(angle)
    if angle != self.zRot:
        self.zRot = angle
        self.emit(QtCore.SIGNAL('zRotationChanged(int)'), angle)
        self.update()