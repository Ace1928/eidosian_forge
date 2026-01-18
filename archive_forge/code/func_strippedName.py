from PySide2 import QtCore, QtGui, QtWidgets
import application_rc
def strippedName(self, fullFileName):
    return QtCore.QFileInfo(fullFileName).fileName()