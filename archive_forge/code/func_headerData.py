from PySide2.QtCore import (Qt, QAbstractTableModel, QModelIndex)
def headerData(self, section, orientation, role=Qt.DisplayRole):
    """ Set the headers to be displayed. """
    if role != Qt.DisplayRole:
        return None
    if orientation == Qt.Horizontal:
        if section == 0:
            return 'Name'
        elif section == 1:
            return 'Address'
    return None