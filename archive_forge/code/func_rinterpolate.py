import pyglet.gl as pgl
from sympy.core import S
def rinterpolate(a_min, a_max, a_value):
    a_range = a_max - a_min
    if a_max == a_min:
        a_range = 1.0
    return (a_value - a_min) / float(a_range)