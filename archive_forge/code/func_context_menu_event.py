import json, os, warnings
from PySide2 import QtCore
from PySide2.QtCore import (QDir, QFileInfo, QModelIndex, QStandardPaths, Qt,
from PySide2.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import (QAction, QDockWidget, QMenu, QMessageBox,
def context_menu_event(self, event):
    context_menu = QMenu()
    open_in_new_tab_action = context_menu.addAction('Open in New Tab')
    remove_action = context_menu.addAction('Remove...')
    current_item = self._current_item()
    open_in_new_tab_action.setEnabled(current_item is not None)
    remove_action.setEnabled(current_item is not None)
    chosen_action = context_menu.exec_(event.globalPos())
    if chosen_action == open_in_new_tab_action:
        self.open_bookmarkInNewTab.emit(current_item.data(_url_role))
    elif chosen_action == remove_action:
        self._remove_item(current_item)