import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def isFinish(self):
    """Returns False if this is the last event in a drag. Note that this
        event will have the same position as the previous one."""
    return self.finish