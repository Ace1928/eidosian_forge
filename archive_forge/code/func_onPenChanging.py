from ..Qt import QtWidgets, QtGui, QtCore
from ..functions import mkPen
def onPenChanging(self, param, val):
    self.pen = QtGui.QPen(val)
    self.update()