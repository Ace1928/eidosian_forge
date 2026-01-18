from PySide2.QtCore import QRect, QRectF, QSize, Qt, QTimer
from PySide2.QtGui import QColor, QPainter, QPalette, QPen
from PySide2.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
def nextAnimationFrame(self):
    self.frameNo += 1
    self.update()