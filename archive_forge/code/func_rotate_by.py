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
def rotate_by(points, radians):
    cx, cy = centroid(points)
    cos = math_cos(radians)
    sin = math_sin(radians)
    newpoints = []
    newpoints_append = newpoints.append
    for i in xrange(0, len(points)):
        qx = (points[i][0] - cx) * cos - (points[i][1] - cy) * sin + cx
        qy = (points[i][0] - cx) * sin + (points[i][1] - cy) * cos + cy
        newpoints_append(Vector(qx, qy))
    return newpoints