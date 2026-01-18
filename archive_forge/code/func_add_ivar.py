import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def add_ivar(self, varname, vartype):
    """Add instance variable named varname to the subclass.
        varname should be a string.
        vartype is a ctypes type.
        The class must be registered AFTER adding instance variables."""
    return add_ivar(self.objc_cls, varname, vartype)