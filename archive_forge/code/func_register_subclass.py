import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def register_subclass(subclass):
    objc.objc_registerClassPair(subclass)