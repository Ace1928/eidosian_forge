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
def delete_selection(self, from_undo=False):
    """Delete the current text selection (if any).
        """
    if self.readonly:
        return
    self._hide_handles(EventLoop.window)
    scroll_x = self.scroll_x
    scroll_y = self.scroll_y
    cc, cr = self.cursor
    if not self._selection:
        return
    text = self.text
    a, b = sorted((self._selection_from, self._selection_to))
    start = self.get_cursor_from_index(a)
    finish = self.get_cursor_from_index(b)
    cur_line = self._lines[start[1]][:start[0]] + self._lines[finish[1]][finish[0]:]
    self._set_line_text(start[1], cur_line)
    start_del, finish_del, lines, lines_flags, len_lines = self._get_line_from_cursor(start[1], cur_line, lines=self._lines[:start[1] + 1] + self._lines[finish[1] + 1:], lines_flags=self._lines_flags[:start[1] + 1] + self._lines_flags[finish[1] + 1:])
    self._refresh_text_from_property('del', start_del, finish_del + (finish[1] - start[1]), lines, lines_flags, len_lines)
    self.scroll_x = scroll_x
    self.scroll_y = scroll_y
    if text[a:b]:
        self._set_unredo_delsel(a, b, text[a:b], from_undo)
    self.cancel_selection()
    self.cursor = self.get_cursor_from_index(a)