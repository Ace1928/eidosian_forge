import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def setYRotation(self, angle):
    angle = self.normalizeAngle(angle)
    if angle != self.yRot:
        self.yRot = angle
        self.emit(QtCore.SIGNAL('yRotationChanged(int)'), angle)
        self.update()