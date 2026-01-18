import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def createColorToolButtonIcon(self, imageFile, color):
    pixmap = QtGui.QPixmap(50, 80)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    image = QtGui.QPixmap(imageFile)
    target = QtCore.QRect(0, 0, 50, 60)
    source = QtCore.QRect(0, 0, 42, 42)
    painter.fillRect(QtCore.QRect(0, 60, 50, 80), color)
    painter.drawPixmap(target, image, source)
    painter.end()
    return QtGui.QIcon(pixmap)