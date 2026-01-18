import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def get_metaclass(name):
    return c_void_p(objc.objc_getMetaClass(ensure_bytes(name)))