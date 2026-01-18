from PySide2 import QtCore, QtGui, QtWidgets, QtPrintSupport
def orderItems(self):
    orderList = []
    for row in range(len(self.items)):
        text = self.itemsTable.item(row, 0).text()
        quantity = int(self.itemsTable.item(row, 1).data(QtCore.Qt.DisplayRole))
        orderList.append((text, max(0, quantity)))
    return orderList