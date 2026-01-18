import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def lineButtonTriggered(self):
    self.scene.setLineColor(QtGui.QColor(self.lineAction.data()))