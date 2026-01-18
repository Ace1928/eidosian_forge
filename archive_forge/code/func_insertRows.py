from PySide2.QtCore import (Qt, QAbstractTableModel, QModelIndex)
def insertRows(self, position, rows=1, index=QModelIndex()):
    """ Insert a row into the model. """
    self.beginInsertRows(QModelIndex(), position, position + rows - 1)
    for row in range(rows):
        self.addresses.insert(position + row, {'name': '', 'address': ''})
    self.endInsertRows()
    return True