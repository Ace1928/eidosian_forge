import OpenGL
import ctypes
from OpenGL import _configflags
from OpenGL import contextdata, error, converters
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,unicode
import logging
from OpenGL import acceleratesupport
def finalise(self, wrapper):
    self.pointerIndex = wrapper.pyArgIndex(self.pointerName)