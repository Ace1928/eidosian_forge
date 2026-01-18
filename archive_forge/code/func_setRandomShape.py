import random
from PySide2 import QtCore, QtGui, QtWidgets
def setRandomShape(self):
    self.setShape(random.randint(1, 7))