from __future__ import print_function
import sys, os
from utils import text_type
from PySide2.QtCore import Property, Signal, Slot, QUrl, Qt
from PySide2.QtGui import QGuiApplication, QPen, QPainter, QColor
from PySide2.QtQml import qmlRegisterType
from PySide2.QtQuick import QQuickPaintedItem, QQuickView
@Slot()
def clearChart(self):
    self.setColor(Qt.transparent)
    self.update()
    self.chartCleared.emit()