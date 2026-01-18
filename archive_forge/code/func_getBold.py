import sys
from os.path import abspath, dirname, join
from PySide2.QtCore import QObject, Slot
from PySide2.QtGui import QGuiApplication
from PySide2.QtQml import QQmlApplicationEngine
@Slot(str, result=bool)
def getBold(self, s):
    if s.lower() == 'bold':
        return True
    else:
        return False