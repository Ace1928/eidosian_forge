import re
import sys
import math
from os import environ
from weakref import ref
from itertools import chain, islice
from kivy.animation import Animation
from kivy.base import EventLoop
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.metrics import inch
from kivy.utils import boundary, platform
from kivy.uix.behaviors import FocusBehavior
from kivy.core.text import Label, DEFAULT_FONT
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix, Callback
from kivy.graphics.context_instructions import Transform
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.bubble import Bubble
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty, \
def get_cursor_from_xy(self, x, y):
    """Return the (col, row) of the cursor from an (x, y) position.
        """
    padding_left, padding_top, padding_right, padding_bottom = self.padding
    lines = self._lines
    dy = self.line_height + self.line_spacing
    cursor_x = x - self.x
    scroll_y = self.scroll_y
    scroll_x = self.scroll_x
    scroll_y = scroll_y / dy if scroll_y > 0 else 0
    cursor_y = self.top - padding_top + scroll_y * dy - y
    cursor_y = int(boundary(round(cursor_y / dy - 0.5), 0, len(lines) - 1))
    get_text_width = self._get_text_width
    tab_width = self.tab_width
    label_cached = self._label_cached
    xoff = 0
    halign = self.halign
    base_dir = self.base_direction or self._resolved_base_dir
    auto_halign_r = halign == 'auto' and base_dir and ('rtl' in base_dir)
    if halign == 'center':
        viewport_width = self.width - padding_left - padding_right
        xoff = max(0, int((viewport_width - self._get_row_width(cursor_y)) / 2))
    elif halign == 'right' or auto_halign_r:
        viewport_width = self.width - padding_left - padding_right
        xoff = max(0, int(viewport_width - self._get_row_width(cursor_y)))
    for i in range(0, len(lines[cursor_y])):
        line_y = lines[cursor_y]
        if cursor_x + scroll_x < xoff + get_text_width(line_y[:i], tab_width, label_cached) + get_text_width(line_y[i], tab_width, label_cached) * 0.6 + padding_left:
            cursor_x = i
            break
    else:
        cursor_x = len(lines[cursor_y])
    return (cursor_x, cursor_y)