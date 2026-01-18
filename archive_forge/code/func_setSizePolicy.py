from ..Qt import QtCore, QtGui, QtWidgets
from .PathButton import PathButton
def setSizePolicy(self, *args, **kwds):
    QtWidgets.QGroupBox.setSizePolicy(self, *args)
    if kwds.pop('closing', False) is True:
        self._lastSizePolicy = self.sizePolicy()