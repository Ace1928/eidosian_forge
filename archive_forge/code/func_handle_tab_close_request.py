from functools import partial
import sys
from bookmarkwidget import BookmarkWidget
from webengineview import WebEngineView
from historywindow import HistoryWindow
from PySide2 import QtCore
from PySide2.QtCore import QPoint, Qt, QUrl
from PySide2.QtWidgets import (QAction, QMenu, QTabBar, QTabWidget)
from PySide2.QtWebEngineWidgets import (QWebEngineDownloadItem,
def handle_tab_close_request(self, index):
    if index >= 0 and self.count() > 1:
        webengineview = self._webengineviews[index]
        if self._history_windows.get(webengineview):
            del self._history_windows[webengineview]
        self._webengineviews.remove(webengineview)
        self.removeTab(index)