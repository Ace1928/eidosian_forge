import sys
import math
import random
from PySide2 import QtCore, QtGui, QtWidgets
def paintBarrier(self, painter):
    painter.setPen(QtCore.Qt.black)
    painter.setBrush(QtCore.Qt.yellow)
    painter.drawRect(self.barrierRect())