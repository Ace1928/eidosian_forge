import pyglet.gl as pgl
from sympy.plotting.pygletplot.plot_rotation import get_spherical_rotatation
from sympy.plotting.pygletplot.util import get_model_matrix, model_to_screen, \
def setup_projection(self):
    pgl.glMatrixMode(pgl.GL_PROJECTION)
    pgl.glLoadIdentity()
    if self.ortho:
        pgl.gluPerspective(0.3, float(self.window.width) / float(self.window.height), self.min_ortho_dist - 0.01, self.max_ortho_dist + 0.01)
    else:
        pgl.gluPerspective(30.0, float(self.window.width) / float(self.window.height), self.min_dist - 0.01, self.max_dist + 0.01)
    pgl.glMatrixMode(pgl.GL_MODELVIEW)