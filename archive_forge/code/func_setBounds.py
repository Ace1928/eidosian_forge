from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine
def setBounds(self, bounds):
    """Set ``(min, max)`` bounding values for the region.

        The current position is only affected it is outside the new bounds. See
        :func:`~pyqtgraph.LinearRegionItem.setRegion` to set the position of the region.

        Use ``(None, None)`` to disable bounds.
        """
    if self.clipItem is not None:
        self.setClipItem(None)
    self._setBounds(bounds)