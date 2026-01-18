import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def removeArrow(self, arrow):
    try:
        self.arrows.remove(arrow)
    except ValueError:
        pass