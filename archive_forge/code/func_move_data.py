import sys
from PySide2.QtCore import (Qt, QByteArray, QModelIndex, QObject, QTimer, QUrl)
from PySide2.QtGui import (QColor, QStandardItemModel, QStandardItem)
from PySide2.QtWidgets import (QApplication, QTreeView)
from PySide2.QtRemoteObjects import (QRemoteObjectHost, QRemoteObjectNode,
def move_data(self):
    self._model.moveRows(QModelIndex(), 2, 4, QModelIndex(), 10)