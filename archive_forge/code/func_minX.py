import random
from PySide2 import QtCore, QtGui, QtWidgets
def minX(self):
    m = self.coords[0][0]
    for i in range(4):
        m = min(m, self.coords[i][0])
    return m