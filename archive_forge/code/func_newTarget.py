import sys
import math
import random
from PySide2 import QtCore, QtGui, QtWidgets
def newTarget(self):
    if CannonField.firstTime:
        CannonField.firstTime = False
        midnight = QtCore.QTime(0, 0, 0)
        random.seed(midnight.secsTo(QtCore.QTime.currentTime()))
    self.target = QtCore.QPoint(200 + random.randint(0, 190 - 1), 10 + random.randint(0, 255 - 1))
    self.update()