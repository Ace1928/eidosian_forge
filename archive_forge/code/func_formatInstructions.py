import sys
import math, random
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtOpenGL import *
def formatInstructions(self, width, height):
    text = self.tr('Click and drag with the left mouse button to rotate the Qt logo.')
    metrics = QFontMetrics(self.font())
    border = max(4, metrics.leading())
    rect = metrics.boundingRect(0, 0, width - 2 * border, int(height * 0.125), Qt.AlignCenter | Qt.TextWordWrap, text)
    self.image = QImage(width, rect.height() + 2 * border, QImage.Format_ARGB32_Premultiplied)
    self.image.fill(qRgba(0, 0, 0, 127))
    painter = QPainter()
    painter.begin(self.image)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.setPen(Qt.white)
    painter.drawText((width - rect.width()) / 2, border, rect.width(), rect.height(), Qt.AlignCenter | Qt.TextWordWrap, text)
    painter.end()