import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
def setEditable(self, editable):
    """
        Set whether this item is user-editable.
        """
    if editable:
        self.setFlags(self.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
    else:
        self.setFlags(self.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)