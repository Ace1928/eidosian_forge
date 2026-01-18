import sys
from PySide2.QtCore import (Qt, QByteArray, QModelIndex, QObject, QTimer, QUrl)
from PySide2.QtGui import (QColor, QStandardItemModel, QStandardItem)
from PySide2.QtWidgets import (QApplication, QTreeView)
from PySide2.QtRemoteObjects import (QRemoteObjectHost, QRemoteObjectNode,
def change_flags(self):
    item = self._model.item(0, 0)
    item.setEnabled(False)
    item = item.child(0, 0)
    item.setFlags(item.flags() & Qt.ItemIsSelectable)