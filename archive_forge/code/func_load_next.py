from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
def load_next(self, mode='next'):
    """Animate to the next slide.

        .. versionadded:: 1.7.0
        """
    if self.index is not None:
        w, h = self.size
        _direction = {'top': -h / 2, 'bottom': h / 2, 'left': w / 2, 'right': -w / 2}
        _offset = _direction[self.direction]
        if mode == 'prev':
            _offset = -_offset
        self._start_animation(min_move=0, offset=_offset)