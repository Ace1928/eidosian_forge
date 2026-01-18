import json, os, warnings
from PySide2 import QtCore
from PySide2.QtCore import (QDir, QFileInfo, QModelIndex, QStandardPaths, Qt,
from PySide2.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import (QAction, QDockWidget, QMenu, QMessageBox,
def add_tool_bar_bookmark(self, url, title, icon):
    self._tool_bar_item().appendRow(_create_item(url, title, icon))