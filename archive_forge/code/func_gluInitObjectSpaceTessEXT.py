from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
def gluInitObjectSpaceTessEXT():
    """Return boolean indicating whether this module is available"""
    return extensions.hasGLUExtension('GLU_EXT_object_space_tess')