from functools import partial
from kivy.animation import Animation
from kivy.compat import string_types
from kivy.config import Config
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.uix.stencilview import StencilView
from kivy.metrics import dp
from kivy.effects.dampedscroll import DampedScrollEffect
from kivy.properties import NumericProperty, BooleanProperty, AliasProperty, \
from kivy.uix.behaviors import FocusBehavior
def convert_distance_to_scroll(self, dx, dy):
    """Convert a distance in pixels to a scroll distance, depending on the
        content size and the scrollview size.

        The result will be a tuple of scroll distance that can be added to
        :data:`scroll_x` and :data:`scroll_y`
        """
    if not self._viewport:
        return (0, 0)
    vp = self._viewport
    if vp.width > self.width:
        sw = vp.width - self.width
        sx = dx / float(sw)
    else:
        sx = 0
    if vp.height > self.height:
        sh = vp.height - self.height
        sy = dy / float(sh)
    else:
        sy = 1
    return (sx, sy)