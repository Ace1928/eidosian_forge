import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def isStart(self):
    """Returns True if this event is the first since a drag was initiated."""
    return self.start