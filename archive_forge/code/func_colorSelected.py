from .. import functions as functions
from ..Qt import QtCore, QtGui, QtWidgets
def colorSelected(self, color):
    self.setColor(self._color, finished=True)