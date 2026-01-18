from ctypes import *
from pyglet.gl.glx import *
from pyglet.util import asstr
def get_server_version(self):
    self.check_display()
    major = c_int()
    minor = c_int()
    if not glXQueryVersion(self.display, byref(major), byref(minor)):
        raise GLXInfoException('Could not determine GLX server version')
    return f'{major.value}.{minor.value}'