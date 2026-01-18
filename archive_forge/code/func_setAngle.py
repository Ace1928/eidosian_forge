import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
@QtCore.Slot(int)
def setAngle(self, angle):
    if angle < 5:
        angle = 5
    if angle > 70:
        angle = 70
    if self.currentAngle == angle:
        return
    self.currentAngle = angle
    self.update()
    self.emit(QtCore.SIGNAL('angleChanged(int)'), self.currentAngle)