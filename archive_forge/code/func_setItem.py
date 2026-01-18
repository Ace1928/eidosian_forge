import sys
from PySide2 import QtCore, QtGui, QtWidgets
def setItem(self):
    items = ('Spring', 'Summer', 'Fall', 'Winter')
    item, ok = QtWidgets.QInputDialog.getItem(self, 'QInputDialog.getItem()', 'Season:', items, 0, False)
    if ok and item:
        self.itemLabel.setText(item)