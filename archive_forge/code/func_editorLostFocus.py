import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def editorLostFocus(self, item):
    cursor = item.textCursor()
    cursor.clearSelection()
    item.setTextCursor(cursor)
    if not item.toPlainText():
        self.removeItem(item)
        item.deleteLater()