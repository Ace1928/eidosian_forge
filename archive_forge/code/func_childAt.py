import sys
from PySide2 import QtCore, QtGui, QtWidgets
def childAt(self, parent, index):
    if parent is not None:
        return parent.child(index)
    else:
        return self.topLevelItem(index)