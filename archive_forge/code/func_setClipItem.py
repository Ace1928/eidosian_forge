from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine
def setClipItem(self, item=None):
    """Set an item to which the region is bounded.

        If ``None``, bounds are disabled.
        """
    self.clipItem = item
    self._clipItemBoundsCache = None
    if item is None:
        self._setBounds((None, None))
    if item is not None:
        self._updateClipItemBounds()