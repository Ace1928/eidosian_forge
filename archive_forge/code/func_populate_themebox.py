import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
def populate_themebox(self):
    theme = self.ui.themeComboBox
    theme.addItem('Light', QtCharts.QChart.ChartThemeLight)
    theme.addItem('Blue Cerulean', QtCharts.QChart.ChartThemeBlueCerulean)
    theme.addItem('Dark', QtCharts.QChart.ChartThemeDark)
    theme.addItem('Brown Sand', QtCharts.QChart.ChartThemeBrownSand)
    theme.addItem('Blue NCS', QtCharts.QChart.ChartThemeBlueNcs)
    theme.addItem('High Contrast', QtCharts.QChart.ChartThemeHighContrast)
    theme.addItem('Blue Icy', QtCharts.QChart.ChartThemeBlueIcy)
    theme.addItem('Qt', QtCharts.QChart.ChartThemeQt)