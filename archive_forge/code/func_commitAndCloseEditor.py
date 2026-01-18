from PySide2.QtWidgets import (QItemDelegate, QStyledItemDelegate, QStyle)
from starrating import StarRating
from stareditor import StarEditor
def commitAndCloseEditor(self):
    """ Erm... commits the data and closes the editor. :) """
    editor = self.sender()
    self.commitData.emit(editor)
    self.closeEditor.emit(editor, QStyledItemDelegate.NoHint)