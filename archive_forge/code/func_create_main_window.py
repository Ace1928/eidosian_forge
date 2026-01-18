import sys
from bookmarkwidget import BookmarkWidget
from browsertabwidget import BrowserTabWidget
from downloadwidget import DownloadWidget
from findtoolbar import FindToolBar
from webengineview import QWebEnginePage, WebEngineView
from PySide2 import QtCore
from PySide2.QtCore import Qt, QUrl
from PySide2.QtGui import QCloseEvent, QKeySequence, QIcon
from PySide2.QtWidgets import (qApp, QAction, QApplication, QDesktopWidget,
from PySide2.QtWebEngineWidgets import (QWebEngineDownloadItem, QWebEnginePage,
def create_main_window():
    """Creates a MainWindow using 75% of the available screen resolution."""
    main_win = MainWindow()
    main_windows.append(main_win)
    available_geometry = app.desktop().availableGeometry(main_win)
    main_win.resize(available_geometry.width() * 2 / 3, available_geometry.height() * 2 / 3)
    main_win.show()
    return main_win