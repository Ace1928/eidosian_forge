import sys
import weakref
import math
from PySide2 import QtCore, QtGui, QtWidgets
def itemMoved(self):
    if not self.timerId:
        self.timerId = self.startTimer(1000 / 25)