from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def get_vectors(self, **kwargs):
    """Return strokes in a format that is acceptable for
        `kivy.multistroke.Recognizer` as a gesture candidate or template. The
        result is cached automatically; the cache is invalidated at the start
        and end of a stroke and if `update_bbox` is called. If you are going
        to analyze a gesture mid-stroke, you may need to set the `no_cache`
        argument to True."""
    if self._cache_time == self._update_time and (not kwargs.get('no_cache')):
        return self._vectors
    vecs = []
    append = vecs.append
    for tuid, l in self._strokes.items():
        lpts = l.points
        append([Vector(*pts) for pts in zip(lpts[::2], lpts[1::2])])
    self._vectors = vecs
    self._cache_time = self._update_time
    return vecs