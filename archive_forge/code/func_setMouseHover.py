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
def setMouseHover(self, hover):
    if self.mouseHovering is hover:
        return
    self.mouseHovering = hover
    if hover:
        self.currentBrush = self.hoverBrush
        self.currentPen = self.hoverPen
    else:
        self.currentBrush = self.brush
        self.currentPen = self.pen
    self.update()