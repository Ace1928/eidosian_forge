from PySide2.QtWidgets import (QItemDelegate, QStyledItemDelegate, QStyle)
from starrating import StarRating
from stareditor import StarEditor
def createEditor(self, parent, option, index):
    """ Creates and returns the custom StarEditor object we'll use to edit
            the StarRating.
        """
    if index.column() == 3:
        editor = StarEditor(parent)
        editor.editingFinished.connect(self.commitAndCloseEditor)
        return editor
    else:
        return QStyledItemDelegate.createEditor(self, parent, option, index)