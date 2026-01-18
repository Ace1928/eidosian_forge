from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def merge_gestures(self, g, other):
    """Merges two gestures together, the oldest one is retained and the
        newer one gets the `GestureContainer.was_merged` flag raised."""
    swap = other._create_time < g._create_time
    a = swap and other or g
    b = swap and g or other
    abbox = a.bbox
    bbbox = b.bbox
    if bbbox['minx'] < abbox['minx']:
        abbox['minx'] = bbbox['minx']
    if bbbox['miny'] < abbox['miny']:
        abbox['miny'] = bbbox['miny']
    if bbbox['maxx'] > abbox['maxx']:
        abbox['maxx'] = bbbox['maxx']
    if bbbox['maxy'] > abbox['maxy']:
        abbox['maxy'] = bbbox['maxy']
    astrokes = a._strokes
    lw = self.line_width
    a_id = a.id
    col = a.color
    self.canvas.remove_group(b.id)
    canv_add = self.canvas.add
    for uid, old in b._strokes.items():
        new_line = Line(points=old.points, width=old.width, group=a_id)
        astrokes[uid] = new_line
        if lw:
            canv_add(Color(col[0], col[1], col[2], mode='rgb', group=a_id))
            canv_add(new_line)
    b.active = False
    b.was_merged = True
    a.active_strokes += b.active_strokes
    a._update_time = Clock.get_time()
    return a