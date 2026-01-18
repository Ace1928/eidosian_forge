import os, sys
from PySide2.QtCore import QDate, QDir, QStandardPaths, Qt, QUrl
from PySide2.QtGui import QClipboard, QGuiApplication, QDesktopServices, QIcon
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import (QAction, qApp, QApplication, QHBoxLayout, QLabel,
from PySide2.QtMultimedia import QCamera, QCameraImageCapture, QCameraInfo
from PySide2.QtMultimediaWidgets import QCameraViewfinder
def takePicture(self):
    self.currentPreview = QImage()
    self.camera.searchAndLock()
    self.imageCapture.capture(self.nextImageFileName())
    self.camera.unlock()