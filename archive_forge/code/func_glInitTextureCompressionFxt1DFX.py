from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.DFX.texture_compression_FXT1 import *
from OpenGL.raw.GL.DFX.texture_compression_FXT1 import _EXTENSION_NAME
def glInitTextureCompressionFxt1DFX():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)