from OpenGL.raw.GL.VERSION import GL_1_1 as _simple
from OpenGL import arrays
from OpenGL import error
from OpenGL import _configflags
import ctypes
def rankPacking(rank):
    """Set the pixel-transfer modes for a given image "rank" (# of dims)
    
    Uses RANK_PACKINGS table to issue calls to glPixelStorei
    """
    for func, which, arg in RANK_PACKINGS[rank]:
        try:
            func(which, arg)
        except error.GLError:
            pass