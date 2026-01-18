from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
def setOn(self, on):
    if self.onVal == on:
        return
    self.onVal = on
    self.update()