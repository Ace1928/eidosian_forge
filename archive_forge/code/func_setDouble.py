import sys
from PySide2 import QtCore, QtGui, QtWidgets
def setDouble(self):
    d, ok = QtWidgets.QInputDialog.getDouble(self, 'QInputDialog.getDouble()', 'Amount:', 37.56, -10000, 10000, 2)
    if ok:
        self.doubleLabel.setText('$%g' % d)