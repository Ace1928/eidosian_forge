import sys
from PySide2.QtCore import Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QPen
from PySide2.QtWidgets import (QApplication, QDoubleSpinBox,
from PySide2.QtCharts import QtCharts
def show_legend_spinbox(self):
    self.legend_settings.setVisible(True)
    chart_viewrect = self.chart_view.rect()
    self.legend_posx.setMinimum(0)
    self.legend_posx.setMaximum(chart_viewrect.width())
    self.legend_posx.setValue(150)
    self.legend_posy.setMinimum(0)
    self.legend_posy.setMaximum(chart_viewrect.height())
    self.legend_posy.setValue(150)
    self.legend_width.setMinimum(0)
    self.legend_width.setMaximum(chart_viewrect.width())
    self.legend_width.setValue(150)
    self.legend_height.setMinimum(0)
    self.legend_height.setMaximum(chart_viewrect.height())
    self.legend_height.setValue(75)