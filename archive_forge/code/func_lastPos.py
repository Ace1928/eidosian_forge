import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def lastPos(self):
    """
        Return the previous position of the mouse in the coordinate system of the item
        that the event was delivered to.
        """
    return Point(self.currentItem.mapFromScene(self._lastScenePos))