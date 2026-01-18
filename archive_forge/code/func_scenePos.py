import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def scenePos(self):
    """Return the current scene position of the mouse."""
    return Point(self._scenePos)