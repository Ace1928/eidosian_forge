import pyglet.gl as pgl
from sympy.plotting.pygletplot.plot_rotation import get_spherical_rotatation
from sympy.plotting.pygletplot.util import get_model_matrix, model_to_screen, \
def mouse_translate(self, x, y, dx, dy):
    pgl.glPushMatrix()
    pgl.glLoadIdentity()
    pgl.glTranslatef(0, 0, -self._dist)
    z = model_to_screen(0, 0, 0)[2]
    d = vec_subs(screen_to_model(x, y, z), screen_to_model(x - dx, y - dy, z))
    pgl.glPopMatrix()
    self._x += d[0]
    self._y += d[1]