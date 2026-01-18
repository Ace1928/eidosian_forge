import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def lastScreenPos(self):
    """Return the previous screen position of the mouse."""
    return Point(self._lastScreenPos)