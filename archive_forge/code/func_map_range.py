import sys
import ctypes
from functools import lru_cache
import pyglet
from pyglet.gl import *
def map_range(self, start, size, ptr_type):
    glBindBuffer(GL_ARRAY_BUFFER, self.id)
    ptr = ctypes.cast(glMapBufferRange(GL_ARRAY_BUFFER, start, size, GL_MAP_WRITE_BIT), ptr_type).contents
    return ptr