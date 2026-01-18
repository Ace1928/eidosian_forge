import sys
from PySide2.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
from PySide2 import QtCore
def createWindow(self, window_type):
    if window_type == QWebEnginePage.WebBrowserTab or window_type == QWebEnginePage.WebBrowserBackgroundTab:
        return self._tab_factory_func()
    return self._window_factory_func()