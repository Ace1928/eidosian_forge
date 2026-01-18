from OpenGL.raw.GL.VERSION import GL_1_1 as _simple
from OpenGL import arrays
from OpenGL import error
from OpenGL import _configflags
import ctypes
def setupDefaultTransferMode():
    """Set pixel transfer mode to assumed internal structure of arrays
    
    Basically OpenGL-ctypes (and PyOpenGL) assume that your image data is in 
    non-byte-swapped order, with big-endian ordering of bytes (though that 
    seldom matters in image data).  These assumptions are normally correct 
    when dealing with Python libraries which expose byte-arrays.
    """
    try:
        _simple.glPixelStorei(_simple.GL_PACK_SWAP_BYTES, 0)
        _simple.glPixelStorei(_simple.GL_PACK_LSB_FIRST, 0)
    except error.GLError:
        pass