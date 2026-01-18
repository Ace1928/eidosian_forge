import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
def createBarChart(self):
    chart = QtCharts.QChart()
    chart.setTitle('Bar chart')
    series = QtCharts.QStackedBarSeries(chart)
    for i in range(len(self.data_table)):
        barset = QtCharts.QBarSet('Bar set {}'.format(i))
        for data in self.data_table[i]:
            barset.append(data[0].y())
        series.append(barset)
    chart.addSeries(series)
    chart.createDefaultAxes()
    chart.axisY().setRange(0, self.value_max * 2)
    chart.axisY().setLabelFormat('%.1f  ')
    return chart