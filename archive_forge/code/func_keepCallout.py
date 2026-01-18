import sys
from PySide2.QtWidgets import (QApplication, QWidget, QGraphicsScene,
from PySide2.QtCore import Qt, QPointF, QRectF, QRect
from PySide2.QtCharts import QtCharts
from PySide2.QtGui import QPainter, QFont, QFontMetrics, QPainterPath, QColor
def keepCallout(self):
    self._callouts.append(self._tooltip)
    self._tooltip = Callout(self._chart)