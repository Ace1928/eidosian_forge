import sys
import math
import random
from PySide2 import QtCore, QtGui, QtWidgets
def targetRect(self):
    result = QtCore.QRect(0, 0, 20, 10)
    result.moveCenter(QtCore.QPoint(self.target.x(), self.height() - 1 - self.target.y()))
    return result