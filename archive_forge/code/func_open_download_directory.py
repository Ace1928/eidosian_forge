import sys
from PySide2 import QtCore
from PySide2.QtCore import QDir, QFileInfo, QStandardPaths, Qt, QUrl
from PySide2.QtGui import QDesktopServices
from PySide2.QtWidgets import (QAction, QLabel, QMenu, QProgressBar,
from PySide2.QtWebEngineWidgets import QWebEngineDownloadItem
@staticmethod
def open_download_directory():
    path = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
    DownloadWidget.open_file(path)