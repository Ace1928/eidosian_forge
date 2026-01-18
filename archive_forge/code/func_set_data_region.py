import sys
import ctypes
from functools import lru_cache
import pyglet
from pyglet.gl import *
def set_data_region(self, data, start, length):
    glBindBuffer(GL_ARRAY_BUFFER, self.id)
    glBufferSubData(GL_ARRAY_BUFFER, start, length, data)