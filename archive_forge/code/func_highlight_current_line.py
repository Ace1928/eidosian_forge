from PySide2.QtCore import Slot, Qt, QRect, QSize
from PySide2.QtGui import QColor, QPainter, QTextFormat
from PySide2.QtWidgets import QPlainTextEdit, QWidget, QTextEdit
@Slot()
def highlight_current_line(self):
    extra_selections = []
    if not self.isReadOnly():
        selection = QTextEdit.ExtraSelection()
        line_color = QColor(Qt.yellow).lighter(160)
        selection.format.setBackground(line_color)
        selection.format.setProperty(QTextFormat.FullWidthSelection, True)
        selection.cursor = self.textCursor()
        selection.cursor.clearSelection()
        extra_selections.append(selection)
    self.setExtraSelections(extra_selections)