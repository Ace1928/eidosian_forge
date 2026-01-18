import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def setItemColor(self, color):
    self.myItemColor = color
    if self.isItemChange(DiagramItem):
        item = self.selectedItems()[0]
        item.setBrush(self.myItemColor)