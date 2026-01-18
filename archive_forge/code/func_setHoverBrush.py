import string
from math import atan2
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .ScatterPlotItem import Symbols
from .TextItem import TextItem
from .UIGraphicsItem import UIGraphicsItem
from .ViewBox import ViewBox
def setHoverBrush(self, *args, **kwargs):
    """Set the brush that fills the symbol when hovering over it. Allowable
        arguments are any that are valid for :func:`~pyqtgraph.mkBrush`.
        """
    self.hoverBrush = fn.mkBrush(*args, **kwargs)
    if self.mouseHovering:
        self.currentBrush = self.hoverBrush
        self.update()