import json, os, warnings
from PySide2 import QtCore
from PySide2.QtCore import (QDir, QFileInfo, QModelIndex, QStandardPaths, Qt,
from PySide2.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import (QAction, QDockWidget, QMenu, QMessageBox,
@staticmethod
def short_title(t):
    i = t.find(' | ')
    if i == -1:
        i = t.find(' - ')
    return t[0:i] if i != -1 else t