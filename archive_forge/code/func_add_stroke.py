from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def add_stroke(self, touch, line):
    """Associate a list of points with a touch.uid; the line itself is
        created by the caller, but subsequent move/up events look it
        up via us. This is done to avoid problems during merge."""
    self._update_time = Clock.get_time()
    self._strokes[str(touch.uid)] = line
    self.active_strokes += 1