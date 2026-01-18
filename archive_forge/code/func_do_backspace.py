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
def do_backspace(self, from_undo=False, mode='bkspc'):
    """Do backspace operation from the current cursor position.
        This action might do several things:

            - removing the current selection if available.
            - removing the previous char and move the cursor back.
            - do nothing, if we are at the start.

        """
    if self.readonly or self._ime_composition:
        return
    col, row = self.cursor
    _lines = self._lines
    _lines_flags = self._lines_flags
    text = _lines[row]
    cursor_index = self.cursor_index()
    if col == 0 and row == 0:
        return
    start = row
    if col == 0:
        if _lines_flags[row] == FL_IS_LINEBREAK:
            substring = u'\n'
            new_text = _lines[row - 1] + text
        else:
            substring = _lines[row - 1][-1] if len(_lines[row - 1]) > 0 else u''
            new_text = _lines[row - 1][:-1] + text
        self._set_line_text(row - 1, new_text)
        self._delete_line(row)
        start = row - 1
    else:
        substring = text[col - 1]
        new_text = text[:col - 1] + text[col:]
        self._set_line_text(row, new_text)
    start, finish, lines, lineflags, len_lines = self._get_line_from_cursor(start, new_text)
    self._refresh_text_from_property('insert' if col == 0 else 'del', start, finish, lines, lineflags, len_lines)
    self.cursor = self.get_cursor_from_index(cursor_index - 1)
    self._set_unredo_bkspc(cursor_index, cursor_index - 1, substring, from_undo, mode)
    self.scroll_x = self.get_max_scroll_x()