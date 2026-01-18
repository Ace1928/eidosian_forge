import copy
import types
from math import *
from PyQt4 import QtCore, QtGui, QtSvg
from rdkit.sping import pid
def stringBox(self, s, font=None):
    """Return the logical width and height of the string if it were drawn     in the current font (defaults to self.font)."""
    if not font:
        font = self.defaultFont
    if font:
        self._adjustFont(font)
    t = QtGui.QGraphicsTextItem(s)
    t.setFont(self._font)
    rect = t.boundingRect()
    return (rect.width(), rect.height())