import sys
from PySide2.QtCore import Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QPen
from PySide2.QtWidgets import (QApplication, QDoubleSpinBox,
from PySide2.QtCharts import QtCharts
def remove_barset(self):
    sets = self.series.barSets()
    len_sets = len(sets)
    if len_sets > 0:
        self.series.remove(sets[len_sets - 1])