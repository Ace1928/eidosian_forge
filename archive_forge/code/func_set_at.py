from collections import namedtuple
from math import floor, ceil
def set_at(surf, in_x, in_y, color):
    """Set the color of a pixel in a surface"""
    surf.set_at((in_x, in_y), color)