import sys
from PySide2.QtCore import Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QPen
from PySide2.QtWidgets import (QApplication, QDoubleSpinBox,
from PySide2.QtCharts import QtCharts
def set_legend_alignment(self):
    button = self.sender()
    legend = self.chart.legend()
    alignment = legend.alignment()
    if alignment == Qt.AlignTop:
        legend.setAlignment(Qt.AlignLeft)
        if button:
            button.setText('Align (Left)')
    elif alignment == Qt.AlignLeft:
        legend.setAlignment(Qt.AlignBottom)
        if button:
            button.setText('Align (Bottom)')
    elif alignment == Qt.AlignBottom:
        legend.setAlignment(Qt.AlignRight)
        if button:
            button.setText('Align (Right)')
    else:
        if button:
            button.setText('Align (Top)')
        legend.setAlignment(Qt.AlignTop)