from ctypes import *
from .base import FontException
import pyglet.lib
def f16p16_to_float(value):
    return float(value) / (1 << 16)