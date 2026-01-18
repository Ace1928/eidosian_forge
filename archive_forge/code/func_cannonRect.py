import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def cannonRect(self):
    result = QtCore.QRect(0, 0, 50, 50)
    result.moveBottomLeft(self.rect().bottomLect())
    return result