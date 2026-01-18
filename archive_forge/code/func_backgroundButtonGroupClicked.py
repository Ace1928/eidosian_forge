import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def backgroundButtonGroupClicked(self, button):
    buttons = self.backgroundButtonGroup.buttons()
    for myButton in buttons:
        if myButton != button:
            button.setChecked(False)
    text = button.text()
    if text == 'Blue Grid':
        self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QPixmap(':/images/background1.png')))
    elif text == 'White Grid':
        self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QPixmap(':/images/background2.png')))
    elif text == 'Gray Grid':
        self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QPixmap(':/images/background3.png')))
    else:
        self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QPixmap(':/images/background4.png')))
    self.scene.update()
    self.view.update()