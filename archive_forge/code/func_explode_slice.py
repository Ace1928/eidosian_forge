import sys
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QPainter
from PySide2.QtWidgets import QApplication, QGridLayout, QWidget
from PySide2.QtCharts import QtCharts
from random import randrange
from functools import partial
def explode_slice(self, exploded, slc):
    if exploded:
        self.update_timer.stop()
        slice_startangle = slc.startAngle()
        slice_endangle = slc.startAngle() + slc.angleSpan()
        donut = slc.series()
        idx = self.donuts.index(donut)
        for i in range(idx + 1, len(self.donuts)):
            self.donuts[i].setPieStartAngle(slice_endangle)
            self.donuts[i].setPieEndAngle(360 + slice_startangle)
    else:
        for donut in self.donuts:
            donut.setPieStartAngle(0)
            donut.setPieEndAngle(360)
        self.update_timer.start()
    slc.setExploded(exploded)