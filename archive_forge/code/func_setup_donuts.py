import sys
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QPainter
from PySide2.QtWidgets import QApplication, QGridLayout, QWidget
from PySide2.QtCharts import QtCharts
from random import randrange
from functools import partial
def setup_donuts(self):
    for i in range(self.donut_count):
        donut = QtCharts.QPieSeries()
        slccount = randrange(3, 6)
        for j in range(slccount):
            value = randrange(100, 200)
            slc = QtCharts.QPieSlice(str(value), value)
            slc.setLabelVisible(True)
            slc.setLabelColor(Qt.white)
            slc.setLabelPosition(QtCharts.QPieSlice.LabelInsideTangential)
            slc.hovered[bool].connect(partial(self.explode_slice, slc=slc))
            donut.append(slc)
            size = (self.max_size - self.min_size) / self.donut_count
            donut.setHoleSize(self.min_size + i * size)
            donut.setPieSize(self.min_size + (i + 1) * size)
        self.donuts.append(donut)
        self.chart_view.chart().addSeries(donut)