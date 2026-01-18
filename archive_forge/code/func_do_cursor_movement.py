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
def do_cursor_movement(self, action, control=False, alt=False):
    """Move the cursor relative to its current position.
        Action can be one of :

            - cursor_left: move the cursor to the left
            - cursor_right: move the cursor to the right
            - cursor_up: move the cursor on the previous line
            - cursor_down: move the cursor on the next line
            - cursor_home: move the cursor at the start of the current line
            - cursor_end: move the cursor at the end of current line
            - cursor_pgup: move one "page" before
            - cursor_pgdown: move one "page" after

        In addition, the behavior of certain actions can be modified:

            - control + cursor_left: move the cursor one word to the left
            - control + cursor_right: move the cursor one word to the right
            - control + cursor_up: scroll up one line
            - control + cursor_down: scroll down one line
            - control + cursor_home: go to beginning of text
            - control + cursor_end: go to end of text
            - alt + cursor_up: shift line(s) up
            - alt + cursor_down: shift line(s) down

        .. versionchanged:: 1.9.1

        """
    if not self._lines:
        return
    col, row = self.cursor
    if action == 'cursor_up':
        result = self._move_cursor_up(col, row, control, alt)
        if result:
            col, row = result
        else:
            return
    elif action == 'cursor_down':
        result = self._move_cursor_down(col, row, control, alt)
        if result:
            col, row = result
        else:
            return
    elif action == 'cursor_home':
        col = 0
        if control:
            row = 0
    elif action == 'cursor_end':
        if control:
            row = len(self._lines) - 1
        col = len(self._lines[row])
    elif action == 'cursor_pgup':
        row = max(0, row - self.pgmove_speed)
        col = min(len(self._lines[row]), col)
    elif action == 'cursor_pgdown':
        row = min(row + self.pgmove_speed, len(self._lines) - 1)
        col = min(len(self._lines[row]), col)
    elif self._selection and self._selection_finished and (self._selection_from < self._selection_to) and (action == 'cursor_left'):
        current_selection_to = self._selection_to
        while self._selection_from != current_selection_to:
            current_selection_to -= 1
            if col:
                col -= 1
            else:
                row -= 1
                col = len(self._lines[row])
    elif self._selection and self._selection_finished and (self._selection_from > self._selection_to) and (action == 'cursor_right'):
        current_selection_to = self._selection_to
        while self._selection_from != current_selection_to:
            current_selection_to += 1
            if len(self._lines[row]) > col:
                col += 1
            else:
                row += 1
                col = 0
    elif action == 'cursor_left':
        if not self.password and control:
            col, row = self._move_cursor_word_left()
        elif col == 0:
            if row:
                row -= 1
                col = len(self._lines[row])
        else:
            col, row = (col - 1, row)
    elif action == 'cursor_right':
        if not self.password and control:
            col, row = self._move_cursor_word_right()
        elif col == len(self._lines[row]):
            if row < len(self._lines) - 1:
                col = 0
                row += 1
        else:
            col, row = (col + 1, row)
    dont_move_cursor = control and action in ['cursor_up', 'cursor_down']
    if dont_move_cursor:
        self._trigger_update_graphics()
    else:
        self.cursor = (col, row)