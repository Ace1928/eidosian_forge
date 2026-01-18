from kivy.event import EventDispatcher
from kivy.eventmanager import (
from kivy.factory import Factory
from kivy.properties import (
from kivy.graphics import (
from kivy.graphics.transformation import Matrix
from kivy.base import EventLoop
from kivy.lang import Builder
from kivy.context import get_current_context
from kivy.weakproxy import WeakProxy
from functools import partial
from itertools import islice
def collide_widget(self, wid):
    """
        Check if another widget collides with this widget. This function
        performs an axis-aligned bounding box intersection test by default.

        :Parameters:
            `wid`: :class:`Widget` class
                Widget to test collision with.

        :Returns:
            bool. True if the other widget collides with this widget, False
            otherwise.

        .. code-block:: python

            >>> wid = Widget(size=(50, 50))
            >>> wid2 = Widget(size=(50, 50), pos=(25, 25))
            >>> wid.collide_widget(wid2)
            True
            >>> wid2.pos = (55, 55)
            >>> wid.collide_widget(wid2)
            False
        """
    if self.right < wid.x:
        return False
    if self.x > wid.right:
        return False
    if self.top < wid.y:
        return False
    if self.y > wid.top:
        return False
    return True