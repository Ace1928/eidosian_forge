from PySide2.QtCore import (Qt, QAbstractTableModel, QModelIndex)
def rowCount(self, index=QModelIndex()):
    """ Returns the number of rows the model holds. """
    return len(self.addresses)