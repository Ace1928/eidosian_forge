from PySide2 import QtCore, QtGui, QtWidgets
def takeAt(self, index):
    if index >= 0 and index < len(self.itemList):
        return self.itemList.pop(index)
    return None