import warnings
from ctypes import c_char_p, cast
from pyglet.gl.gl import GL_EXTENSIONS, GL_RENDERER, GL_VENDOR, GL_VERSION
from pyglet.gl.gl import GL_MAJOR_VERSION, GL_MINOR_VERSION, GLint
from pyglet.gl.lib import GLException
from pyglet.util import asstr
def remove_active_context(self):
    self._have_context = False
    self._have_info = False