from PySide2.QtCore import (Qt, QAbstractTableModel, QModelIndex)
def removeRows(self, position, rows=1, index=QModelIndex()):
    """ Remove a row from the model. """
    self.beginRemoveRows(QModelIndex(), position, position + rows - 1)
    del self.addresses[position:position + rows]
    self.endRemoveRows()
    return True