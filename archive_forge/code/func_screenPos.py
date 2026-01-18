import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def screenPos(self):
    """Return the current screen position of the mouse."""
    return Point(self._screenPos)