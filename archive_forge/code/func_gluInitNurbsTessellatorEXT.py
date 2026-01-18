from OpenGL import extensions
from OpenGL.raw.GL import _types
from OpenGL.raw.GLU import constants
def gluInitNurbsTessellatorEXT():
    """Return boolean indicating whether this module is available"""
    return extensions.hasGLUExtension('GLU_EXT_nurbs_tessellator')