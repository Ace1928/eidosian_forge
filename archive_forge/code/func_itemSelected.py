import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def itemSelected(self, item):
    font = item.font()
    color = item.defaultTextColor()
    self.fontCombo.setCurrentFont(font)
    self.fontSizeCombo.setEditText(str(font.pointSize()))
    self.boldAction.setChecked(font.weight() == QtGui.QFont.Bold)
    self.italicAction.setChecked(font.italic())
    self.underlineAction.setChecked(font.underline())