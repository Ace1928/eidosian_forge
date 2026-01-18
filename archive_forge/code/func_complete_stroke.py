from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def complete_stroke(self):
    """Called on touch up events to keep track of how many strokes
        are active in the gesture (we only want to dispatch event when
        the *last* stroke in the gesture is released)"""
    self._update_time = Clock.get_time()
    self.active_strokes -= 1