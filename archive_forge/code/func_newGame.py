import sys
import math
import random
from PySide2 import QtCore, QtGui, QtWidgets
@QtCore.Slot()
def newGame(self):
    self.shotsLeft.display(15)
    self.hits.display(0)
    self.cannonField.restartGame()
    self.cannonField.newTarget()