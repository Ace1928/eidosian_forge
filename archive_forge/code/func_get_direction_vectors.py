import pyglet.gl as pgl
from sympy.core import S
def get_direction_vectors():
    m = get_model_matrix()
    return ((m[0], m[4], m[8]), (m[1], m[5], m[9]), (m[2], m[6], m[10]))