import pyglet.gl as pgl
from sympy.core import S
def scale_value(v, v_min, v_len):
    return (v - v_min) / v_len