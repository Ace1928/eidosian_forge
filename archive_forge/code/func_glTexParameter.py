from OpenGL import arrays
from OpenGL.arrays.arraydatatype import GLfloatArray
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL.GL.VERSION import GL_1_1 as full
from OpenGL.raw.GL import _errors
from OpenGL._bytes import bytes
from OpenGL import _configflags
from OpenGL._null import NULL as _NULL
import ctypes
def glTexParameter(target, pname, parameter):
    """Set a texture parameter, choose underlying call based on pname and parameter"""
    if isinstance(parameter, float):
        return full.glTexParameterf(target, pname, parameter)
    elif isinstance(parameter, int):
        return full.glTexParameteri(target, pname, parameter)
    else:
        value = GLfloatArray.asArray(parameter, full.GL_FLOAT)
        return full.glTexParameterfv(target, pname, value)