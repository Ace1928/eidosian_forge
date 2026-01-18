import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
@QtCore.Slot(int)
def setForce(self, force):
    if force < 0:
        force = 0
    if self.currentForce == force:
        return
    self.currentForce = force
    self.emit(QtCore.SIGNAL('forceChanged(int)'), self.currentForce)