from OpenGL import arrays
from OpenGL.arrays.arraydatatype import GLfloatArray
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL.GL.VERSION import GL_1_1 as full
from OpenGL.raw.GL import _errors
from OpenGL._bytes import bytes
from OpenGL import _configflags
from OpenGL._null import NULL as _NULL
import ctypes
def glMap1(target, u1, u2, points):
    """glMap1(target, u1, u2, points[][][]) -> None

        This is a completely non-standard signature which doesn't allow for most
        of the funky uses with strides and the like, but it has been like this for
        a very long time...
        """
    ptr = arrayType.asArray(points)
    dims = arrayType.dimensions(ptr)
    uorder = dims[0]
    ustride = dims[1]
    return baseFunction(target, u1, u2, ustride, uorder, ptr)