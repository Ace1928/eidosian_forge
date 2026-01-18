import pyglet.gl as pgl
from sympy.plotting.pygletplot.plot_rotation import get_spherical_rotatation
from sympy.plotting.pygletplot.util import get_model_matrix, model_to_screen, \
def euler_rotate(self, angle, x, y, z):
    pgl.glPushMatrix()
    pgl.glLoadMatrixf(self._rot)
    pgl.glRotatef(angle, x, y, z)
    self._rot = get_model_matrix()
    pgl.glPopMatrix()