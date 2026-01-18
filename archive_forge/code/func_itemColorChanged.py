import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def itemColorChanged(self):
    self.fillAction = self.sender()
    self.fillColorToolButton.setIcon(self.createColorToolButtonIcon(':/images/floodfill.png', QtGui.QColor(self.fillAction.data())))
    self.fillButtonTriggered()