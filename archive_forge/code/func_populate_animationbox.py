import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
def populate_animationbox(self):
    animated = self.ui.animatedComboBox
    animated.addItem('No Animations', QtCharts.QChart.NoAnimation)
    animated.addItem('GridAxis Animations', QtCharts.QChart.GridAxisAnimations)
    animated.addItem('Series Animations', QtCharts.QChart.SeriesAnimations)
    animated.addItem('All Animations', QtCharts.QChart.AllAnimations)