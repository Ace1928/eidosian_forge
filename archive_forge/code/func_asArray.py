import operator
from OpenGL.arrays import buffers
from OpenGL.raw.GL import _types 
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL import constant, error
@classmethod
def asArray(cls, value, typeCode=None):
    """Convert given value to an array value of given typeCode"""
    return super(NumpyHandler, cls).asArray(cls.contiguous(value, typeCode), typeCode)