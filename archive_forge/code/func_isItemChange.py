import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def isItemChange(self, type):
    for item in self.selectedItems():
        if isinstance(item, type):
            return True
    return False