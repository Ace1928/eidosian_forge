import pyglet.gl as pgl
from sympy.plotting.pygletplot.plot_rotation import get_spherical_rotatation
from sympy.plotting.pygletplot.util import get_model_matrix, model_to_screen, \
def spherical_rotate(self, p1, p2, sensitivity=1.0):
    mat = get_spherical_rotatation(p1, p2, self.window.width, self.window.height, sensitivity)
    if mat is not None:
        self.mult_rot_matrix(mat)