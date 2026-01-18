import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def buttonDownScenePos(self, btn=None):
    """
        Return the scene position of the mouse at the time *btn* was pressed.
        If *btn* is omitted, then the button that initiated the drag is assumed.
        """
    if btn is None:
        btn = self.button()
    return Point(self._buttonDownScenePos[btn])