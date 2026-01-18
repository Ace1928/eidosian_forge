import sys
import math
import random
from PySide2 import QtCore, QtGui, QtWidgets
def paintTarget(self, painter):
    painter.setPen(QtCore.Qt.black)
    painter.setBrush(QtCore.Qt.red)
    painter.drawRect(self.targetRect())