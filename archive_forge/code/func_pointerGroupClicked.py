import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def pointerGroupClicked(self, i):
    self.scene.setMode(self.pointerTypeGroup.checkedId())