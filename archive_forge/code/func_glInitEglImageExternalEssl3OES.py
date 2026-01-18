from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.OES.EGL_image_external_essl3 import *
from OpenGL.raw.GLES2.OES.EGL_image_external_essl3 import _EXTENSION_NAME
def glInitEglImageExternalEssl3OES():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)