import random
from PySide2 import QtCore, QtGui, QtWidgets
def maxX(self):
    m = self.coords[0][0]
    for i in range(4):
        m = max(m, self.coords[i][0])
    return m