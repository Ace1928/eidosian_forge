import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
def create_areachart(self):
    chart = QtCharts.QChart()
    chart.setTitle('Area Chart')
    lower_series = None
    name = 'Series '
    for i in range(len(self.data_table)):
        upper_series = QtCharts.QLineSeries(chart)
        for j in range(len(self.data_table[i])):
            data = self.data_table[i][j]
            if lower_series:
                points = lower_series.pointsVector()
                y_value = points[i].y() + data[0].y()
                upper_series.append(QPointF(j, y_value))
            else:
                upper_series.append(QPointF(j, data[0].y()))
        area = QtCharts.QAreaSeries(upper_series, lower_series)
        area.setName('{}{}'.format(name, i))
        chart.addSeries(area)
        lower_series = upper_series
    chart.createDefaultAxes()
    chart.axisX().setRange(0, self.value_count - 1)
    chart.axisY().setRange(0, self.value_max)
    chart.axisY().setLabelFormat('%.1f  ')
    return chart