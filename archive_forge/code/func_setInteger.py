import sys
from PySide2 import QtCore, QtGui, QtWidgets
def setInteger(self):
    i, ok = QtWidgets.QInputDialog.getInt(self, 'QInputDialog.getInteger()', 'Percentage:', 25, 0, 100, 1)
    if ok:
        self.integerLabel.setText('%d%%' % i)