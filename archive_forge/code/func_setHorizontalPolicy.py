from ..Qt import QtCore, QtGui, QtWidgets
from .PathButton import PathButton
def setHorizontalPolicy(self, *args):
    QtWidgets.QGroupBox.setHorizontalPolicy(self, *args)
    self._lastSizePolicy = self.sizePolicy()