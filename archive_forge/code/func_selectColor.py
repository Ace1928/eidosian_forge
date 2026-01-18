from .. import functions as functions
from ..Qt import QtCore, QtGui, QtWidgets
def selectColor(self):
    self.origColor = self.color()
    self.colorDialog.setCurrentColor(self.color())
    self.colorDialog.open()