import sys
from PySide2 import QtCore, QtGui, QtWidgets
def maybeRefresh(self):
    if self.state() != QtWidgets.QAbstractItemView.EditingState:
        self.refresh()