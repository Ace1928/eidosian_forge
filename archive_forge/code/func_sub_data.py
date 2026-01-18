import sys
import ctypes
from functools import lru_cache
import pyglet
from pyglet.gl import *
def sub_data(self):
    """Updates the buffer if any data has been changed or invalidated. Allows submitting multiple changes at once,
        rather than having to call glBufferSubData for every change."""
    if not self._dirty:
        return
    glBindBuffer(GL_ARRAY_BUFFER, self.id)
    size = self._dirty_max - self._dirty_min
    if size > 0:
        if size == self.size:
            glBufferData(GL_ARRAY_BUFFER, self.size, self.data, self.usage)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, self._dirty_min, size, self.data_ptr + self._dirty_min)
        self._dirty_min = sys.maxsize
        self._dirty_max = 0
        self._dirty = False