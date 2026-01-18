from math import cos, sin, pi, sqrt, atan
from colorsys import rgb_to_hsv, hsv_to_rgb
from kivy.clock import Clock
from kivy.graphics import Mesh, InstructionGroup, Color
from kivy.logger import Logger
from kivy.properties import (NumericProperty, BoundedNumericProperty,
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex, get_hex_from_color
def rect_to_polar(origin, x, y):
    if x == origin[0]:
        if y == origin[1]:
            return (0, 0)
        elif y > origin[1]:
            return (y - origin[1], pi / 2)
        else:
            return (origin[1] - y, 3 * pi / 2)
    t = atan(float(y - origin[1]) / (x - origin[0]))
    if x - origin[0] < 0:
        t += pi
    if t < 0:
        t += 2 * pi
    return (distance((x, y), origin), t)