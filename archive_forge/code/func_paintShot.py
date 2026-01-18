import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def paintShot(self, painter):
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtCore.Qt.black)
    painter.drawRect(self.shotRect())