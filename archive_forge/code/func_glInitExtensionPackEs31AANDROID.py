from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.ANDROID.extension_pack_es31a import *
from OpenGL.raw.GLES2.ANDROID.extension_pack_es31a import _EXTENSION_NAME
def glInitExtensionPackEs31AANDROID():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)