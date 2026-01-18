from ..Qt import QtCore, QtWidgets
def setColumnCount(self, c):
    QtWidgets.QTreeWidget.setColumnCount(self, c)
    self.sigColumnCountChanged.emit(self, c)