from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ARB.sync import *
from OpenGL.raw.GL.ARB.sync import _EXTENSION_NAME
from OpenGL.raw.GL._types import GLint
from OpenGL.arrays import GLintArray
def glGetSync(sync, pname, bufSize=1, length=None, values=None):
    """Wrapper around glGetSynciv that auto-allocates buffers
    
    sync -- the GLsync struct pointer (see glGetSynciv)
    pname -- constant to retrieve (see glGetSynciv)
    bufSize -- defaults to 1, maximum number of items to retrieve,
        currently all constants are defined to return a single 
        value 
    length -- None or a GLint() instance (ONLY!), must be a byref()
        capable object with a .value attribute which retrieves the 
        set value
    values -- None or an array object, if None, will be a default 
        return-array-type of length bufSize
    
    returns values[:length.value], i.e. an array with the values set 
    by the call, currently always a single-value array.
    """
    if values is None:
        values = GLintArray.zeros((bufSize,))
    if length is None:
        length = GLint()
    glGetSynciv(sync, pname, bufSize, length, values)
    written = length.value
    return values[:written]