from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.NV.shader_storage_buffer_object import *
from OpenGL.raw.GL.NV.shader_storage_buffer_object import _EXTENSION_NAME
def glInitShaderStorageBufferObjectNV():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)