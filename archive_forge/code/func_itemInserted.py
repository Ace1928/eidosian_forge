import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def itemInserted(self, item):
    self.pointerTypeGroup.button(DiagramScene.MoveItem).setChecked(True)
    self.scene.setMode(self.pointerTypeGroup.checkedId())
    self.buttonGroup.button(item.diagramType).setChecked(False)