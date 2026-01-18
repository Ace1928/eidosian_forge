from PySide2 import QtCore, QtGui, QtWidgets
def itemAt(self, index):
    if index >= 0 and index < len(self.itemList):
        return self.itemList[index]
    return None