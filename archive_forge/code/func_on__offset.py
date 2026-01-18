from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
def on__offset(self, *args):
    self._trigger_position_visible_slides()
    direction = self.direction[0]
    _offset = self._offset
    width = self.width
    height = self.height
    index = self.index
    if self._skip_slide is not None or index is None:
        return
    if direction == 'r' and _offset <= -width or (direction == 'l' and _offset >= width) or (direction == 't' and _offset <= -height) or (direction == 'b' and _offset >= height):
        if self.next_slide:
            self.index += 1
    elif direction == 'r' and _offset >= width or (direction == 'l' and _offset <= -width) or (direction == 't' and _offset >= height) or (direction == 'b' and _offset <= -height):
        if self.previous_slide:
            self.index -= 1
    elif self._prev_equals_next:
        new_value = (_offset < 0) is (direction in 'rt')
        if self._prioritize_next is not new_value:
            self._prioritize_next = new_value
            if new_value is (self._next is None):
                self._prev, self._next = (self._next, self._prev)