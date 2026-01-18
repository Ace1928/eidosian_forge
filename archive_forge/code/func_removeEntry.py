from PySide2.QtCore import (Qt, Signal, QRegExp, QModelIndex,
from PySide2.QtWidgets import (QWidget, QTabWidget, QMessageBox, QTableView,
from tablemodel import TableModel
from newaddresstab import NewAddressTab
from adddialogwidget import AddDialogWidget
def removeEntry(self):
    """ Remove an entry from the addressbook. """
    tableView = self.currentWidget()
    proxyModel = tableView.model()
    selectionModel = tableView.selectionModel()
    indexes = selectionModel.selectedRows()
    for index in indexes:
        row = proxyModel.mapToSource(index).row()
        self.tableModel.removeRows(row)
    if self.tableModel.rowCount() == 0:
        self.insertTab(0, self.newAddressTab, 'Address Book')