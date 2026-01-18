from .. import functions as fn
from .. import getConfigOption
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
def setAntialiasing(self, aa):
    """Enable or disable default antialiasing.
        Note that this will only affect items that do not specify their own antialiasing options."""
    if aa:
        self.setRenderHints(self.renderHints() | QtGui.QPainter.RenderHint.Antialiasing)
    else:
        self.setRenderHints(self.renderHints() & ~QtGui.QPainter.RenderHint.Antialiasing)