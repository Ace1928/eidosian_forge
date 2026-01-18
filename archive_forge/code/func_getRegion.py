from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine
def getRegion(self):
    """Return the values at the edges of the region."""
    r = (self.lines[0].value(), self.lines[1].value())
    if self.swapMode == 'sort':
        return (min(r), max(r))
    else:
        return r