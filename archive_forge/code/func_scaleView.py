import sys
import weakref
import math
from PySide2 import QtCore, QtGui, QtWidgets
def scaleView(self, scaleFactor):
    factor = self.matrix().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
    if factor < 0.07 or factor > 100:
        return
    self.scale(scaleFactor, scaleFactor)