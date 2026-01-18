from pyglet.window import key
from pyglet.window.mouse import LEFT, RIGHT, MIDDLE
from sympy.plotting.pygletplot.util import get_direction_vectors, get_basis_vectors
def is_2D(self):
    functions = self.window.plot._functions
    for i in functions:
        if len(functions[i].i_vars) > 1 or len(functions[i].d_vars) > 2:
            return False
    return True