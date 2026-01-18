import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def lastScenePos(self):
    """Return the previous scene position of the mouse."""
    return Point(self._lastScenePos)