import sys
from PySide2 import QtCore, QtGui, QtWidgets
def moveItemForward(self, parent, oldIndex, newIndex):
    for int in range(oldIndex - newIndex):
        self.deleteItem(parent, newIndex)