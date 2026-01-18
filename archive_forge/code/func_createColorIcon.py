import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def createColorIcon(self, color):
    pixmap = QtGui.QPixmap(20, 20)
    painter = QtGui.QPainter(pixmap)
    painter.setPen(QtCore.Qt.NoPen)
    painter.fillRect(QtCore.QRect(0, 0, 20, 20), color)
    painter.end()
    return QtGui.QIcon(pixmap)