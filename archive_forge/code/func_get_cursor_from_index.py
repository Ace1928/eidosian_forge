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
def get_cursor_from_index(self, index):
    """Return the (col, row) of the cursor from text index.
        """
    index = boundary(index, 0, len(self.text))
    if index <= 0:
        return (0, 0)
    flags = self._lines_flags
    lines = self._lines
    if not lines:
        return (0, 0)
    i = 0
    for row, line in enumerate(lines):
        count = i + len(line)
        if flags[row] & FL_IS_LINEBREAK:
            count += 1
            i += 1
        if count >= index:
            return (index - i, row)
        i = count
    return (int(index), int(row))