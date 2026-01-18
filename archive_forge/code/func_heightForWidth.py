from PySide2 import QtCore, QtGui, QtWidgets
def heightForWidth(self, width):
    height = self.doLayout(QtCore.QRect(0, 0, width, 0), True)
    return height