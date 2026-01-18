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
def get_protractor_vector(self, numpoints, orientation_sens):
    """(Internal use only) Return vector for comparing to a
        UnistrokeTemplate with Protractor"""
    return self._get_db_key('vector', numpoints, orientation_sens)