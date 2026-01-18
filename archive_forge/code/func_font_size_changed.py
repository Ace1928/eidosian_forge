import sys
from PySide2.QtCore import Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QPen
from PySide2.QtWidgets import (QApplication, QDoubleSpinBox,
from PySide2.QtCharts import QtCharts
def font_size_changed(self):
    legend = self.chart.legend()
    font = legend.font()
    font_size = self.font_size.value()
    if font_size < 1:
        font_size = 1
    font.setPointSizeF(font_size)
    legend.setFont(font)