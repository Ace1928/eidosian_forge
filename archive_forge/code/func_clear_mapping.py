import sys
from random import randrange
from PySide2.QtCore import QAbstractTableModel, QModelIndex, QRect, Qt
from PySide2.QtGui import QColor, QPainter
from PySide2.QtWidgets import (QApplication, QGridLayout, QHeaderView,
from PySide2.QtCharts import QtCharts
def clear_mapping(self):
    self.mapping = {}