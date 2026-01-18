import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def buttonGroupClicked(self, id):
    buttons = self.buttonGroup.buttons()
    for button in buttons:
        if self.buttonGroup.button(id) != button:
            button.setChecked(False)
    if id == self.InsertTextButton:
        self.scene.setMode(DiagramScene.InsertText)
    else:
        self.scene.setItemType(id)
        self.scene.setMode(DiagramScene.InsertItem)