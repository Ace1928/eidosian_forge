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
def select_text(self, start, end):
    """ Select a portion of text displayed in this TextInput.

        .. versionadded:: 1.4.0

        :Parameters:
            `start`
                Index of textinput.text from where to start selection
            `end`
                Index of textinput.text till which the selection should be
                displayed
        """
    if end < start:
        raise Exception('end must be superior to start')
    text_length = len(self.text)
    self._selection_from = boundary(start, 0, text_length)
    self._selection_to = boundary(end, 0, text_length)
    self._selection_finished = True
    self._update_selection(True)
    self._update_graphics_selection()