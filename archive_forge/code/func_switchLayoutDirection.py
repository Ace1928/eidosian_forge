from PySide2.QtCore import (QFile, QFileInfo, QPoint, QSettings, QSignalMapper,
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
import mdi_rc
def switchLayoutDirection(self):
    if self.layoutDirection() == Qt.LeftToRight:
        QApplication.setLayoutDirection(Qt.RightToLeft)
    else:
        QApplication.setLayoutDirection(Qt.LeftToRight)