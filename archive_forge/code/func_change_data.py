import sys
from PySide2.QtCore import (Qt, QByteArray, QModelIndex, QObject, QTimer, QUrl)
from PySide2.QtGui import (QColor, QStandardItemModel, QStandardItem)
from PySide2.QtWidgets import (QApplication, QTreeView)
from PySide2.QtRemoteObjects import (QRemoteObjectHost, QRemoteObjectNode,
def change_data(self):
    for i in range(10, 50):
        self._model.setData(self._model.index(i, 1), QColor(Qt.blue), Qt.BackgroundRole)