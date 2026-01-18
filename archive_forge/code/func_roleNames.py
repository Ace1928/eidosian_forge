from __future__ import print_function
import os
import sys
import PySide2.QtQml
from PySide2.QtCore import QAbstractListModel, Qt, QUrl, QByteArray
from PySide2.QtGui import QGuiApplication
from PySide2.QtQuick import QQuickView
def roleNames(self):
    roles = {PersonModel.MyRole: QByteArray(b'modelData'), Qt.DisplayRole: QByteArray(b'display')}
    return roles