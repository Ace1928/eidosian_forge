from ctypes import *
from pyglet.gl.glx import *
from pyglet.util import asstr
def check_display(self):
    if not self.display:
        raise GLXInfoException('No X11 display has been set yet.')