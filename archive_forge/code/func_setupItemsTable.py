from PySide2 import QtCore, QtGui, QtWidgets, QtPrintSupport
def setupItemsTable(self):
    self.itemsTable = QtWidgets.QTableWidget(len(self.items), 2)
    for row, item in enumerate(self.items):
        name = QtWidgets.QTableWidgetItem(item)
        name.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        self.itemsTable.setItem(row, 0, name)
        quantity = QtWidgets.QTableWidgetItem('1')
        self.itemsTable.setItem(row, 1, quantity)