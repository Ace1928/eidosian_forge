from OpenGL import arrays
from OpenGL.arrays.arraydatatype import GLfloatArray
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL.GL.VERSION import GL_1_1 as full
from OpenGL.raw.GL import _errors
from OpenGL._bytes import bytes
from OpenGL import _configflags
from OpenGL._null import NULL as _NULL
import ctypes
def glRasterPos(*args):
    """Choose glRasterPosX based on number of args"""
    if len(args) == 1:
        args = args[0]
    function = glRasterPosDispatch[len(args)]
    return function(*args)