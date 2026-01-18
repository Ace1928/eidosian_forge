import random
from PySide2 import QtCore, QtGui, QtWidgets
def minY(self):
    m = self.coords[0][1]
    for i in range(4):
        m = min(m, self.coords[i][1])
    return m