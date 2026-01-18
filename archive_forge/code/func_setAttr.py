from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets, QtGui
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
def setAttr(self, attr, value):
    """Set default text properties. See setText() for accepted parameters."""
    self.opts[attr] = value