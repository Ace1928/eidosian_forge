import sys
from PySide2.QtCore import SLOT, QStandardPaths, Qt
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, qApp, QApplication, QDialog, QFileDialog,
from PySide2.QtMultimedia import QMediaPlayer, QMediaPlaylist
from PySide2.QtMultimediaWidgets import QVideoWidget
def previousClicked(self):
    if self.player.position() <= 5000:
        self.playlist.previous()
    else:
        player.setPosition(0)