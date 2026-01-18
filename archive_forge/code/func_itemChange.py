import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def itemChange(self, change, value):
    if change == QtWidgets.QGraphicsItem.ItemPositionChange:
        for arrow in self.arrows:
            arrow.updatePosition()
    return value