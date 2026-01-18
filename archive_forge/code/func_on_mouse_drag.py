from pyglet.window import key
from pyglet.window.mouse import LEFT, RIGHT, MIDDLE
from sympy.plotting.pygletplot.util import get_direction_vectors, get_basis_vectors
def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
    if buttons & LEFT:
        if self.is_2D():
            self.camera.mouse_translate(x, y, dx, dy)
        else:
            self.camera.spherical_rotate((x - dx, y - dy), (x, y), self.get_mouse_sensitivity())
    if buttons & MIDDLE:
        self.camera.zoom_relative([1, -1][self.invert_mouse_zoom] * dy, self.get_mouse_sensitivity() / 20.0)
    if buttons & RIGHT:
        self.camera.mouse_translate(x, y, dx, dy)