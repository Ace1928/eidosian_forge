import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def isExit(self):
    """Returns True if the mouse has just exited the item's shape"""
    return self.exit