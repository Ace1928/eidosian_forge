import os
from os.path import dirname, join, exists, abspath
from kivy.clock import Clock
from kivy.compat import PY2
from kivy.properties import ObjectProperty, NumericProperty, \
from kivy.lang import Builder
from kivy.utils import get_hex_from_color, get_color_from_hex
from kivy.uix.widget import Widget
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import AsyncImage, Image
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.anchorlayout import AnchorLayout
from kivy.animation import Animation
from kivy.logger import Logger
from docutils.parsers import rst
from docutils.parsers.rst import roles
from docutils import nodes, frontend, utils
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.roles import set_classes
def goto(self, ref, *largs):
    """Scroll to the reference. If it's not found, nothing will be done.

        For this text::

            .. _myref:

            This is something I always wanted.

        You can do::

            from kivy.clock import Clock
            from functools import partial

            doc = RstDocument(...)
            Clock.schedule_once(partial(doc.goto, 'myref'), 0.1)

        .. note::

            It is preferable to delay the call of the goto if you just loaded
            the document because the layout might not be finished or the
            size of the RstDocument has not yet been determined. In
            either case, the calculation of the scrolling would be
            wrong.

            You can, however, do a direct call if the document is already
            loaded.

        .. versionadded:: 1.3.0
        """
    if ref.endswith('.rst'):
        self.source = ref
        return
    ref = self.refs_assoc.get(ref, ref)
    ax = ay = None
    for node in self.anchors_widgets:
        if ref in node.anchors:
            ax, ay = node.anchors[ref]
            break
    if ax is None:
        return
    ax += node.x
    ay = node.top - ay
    sx, sy = (self.scatter.x, self.scatter.top)
    ay -= self.height
    dx, dy = self.convert_distance_to_scroll(0, ay)
    dy = max(0, min(1, dy))
    Animation(scroll_y=dy, d=0.25, t='in_out_expo').start(self)