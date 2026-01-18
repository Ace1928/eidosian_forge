import sys
from PySide2 import QtCore, QtGui, QtWidgets
def setFallbacksEnabled(self, enabled):
    if self.settings is not None:
        self.settings.setFallbacksEnabled(enabled)
        self.refresh()