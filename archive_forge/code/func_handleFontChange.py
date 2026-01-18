import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def handleFontChange(self):
    font = self.fontCombo.currentFont()
    font.setPointSize(int(self.fontSizeCombo.currentText()))
    if self.boldAction.isChecked():
        font.setWeight(QtGui.QFont.Bold)
    else:
        font.setWeight(QtGui.QFont.Normal)
    font.setItalic(self.italicAction.isChecked())
    font.setUnderline(self.underlineAction.isChecked())
    self.scene.setFont(font)