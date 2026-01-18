from __future__ import print_function
import sys, os
from utils import text_type
from PySide2.QtCore import Property, QUrl
from PySide2.QtGui import QGuiApplication, QPen, QPainter, QColor
from PySide2.QtQml import qmlRegisterType, ListProperty
from PySide2.QtQuick import QQuickPaintedItem, QQuickView, QQuickItem
def getAngleSpan(self):
    return self._angleSpan