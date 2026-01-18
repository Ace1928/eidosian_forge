import pickle
import base64
import zlib
from re import match as re_match
from collections import deque
from math import sqrt, pi, radians, acos, atan, atan2, pow, floor
from math import sin as math_sin, cos as math_cos
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.properties import ListProperty
from kivy.compat import PY2
from io import BytesIO
def parse_gesture(self, data):
    """Parse data formatted by export_gesture(). Returns a list of
        :class:`MultistrokeGesture` objects. This is used internally by
        :meth:`import_gesture`, you normally don't need to call this
        directly."""
    io = BytesIO(zlib.decompress(base64.b64decode(data)))
    p = pickle.Unpickler(io)
    multistrokes = []
    ms_append = multistrokes.append
    for multistroke in p.load():
        strokes = multistroke['strokes']
        multistroke['strokes'] = [[Vector(x, y) for x, y in line] for line in strokes]
        ms_append(MultistrokeGesture(**multistroke))
    return multistrokes