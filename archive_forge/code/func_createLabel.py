from PySide2.QtCore import QRect, QRectF, QSize, Qt, QTimer
from PySide2.QtGui import QColor, QPainter, QPalette, QPen
from PySide2.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
def createLabel(self, text):
    label = QLabel(text)
    label.setAlignment(Qt.AlignCenter)
    label.setMargin(2)
    label.setFrameStyle(QFrame.Box | QFrame.Sunken)
    return label