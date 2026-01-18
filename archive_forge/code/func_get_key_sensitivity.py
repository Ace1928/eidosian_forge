from pyglet.window import key
from pyglet.window.mouse import LEFT, RIGHT, MIDDLE
from sympy.plotting.pygletplot.util import get_direction_vectors, get_basis_vectors
def get_key_sensitivity(self):
    if self.action['modify_sensitivity']:
        return self.modified_key_sensitivity
    else:
        return self.normal_key_sensitivity