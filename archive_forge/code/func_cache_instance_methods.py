import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def cache_instance_methods(self):
    """Create and store python representations of all instance methods
        implemented by this class (but does not find methods of superclass)."""
    count = c_uint()
    method_array = objc.class_copyMethodList(self.ptr, byref(count))
    for i in range(count.value):
        method = c_void_p(method_array[i])
        objc_method = ObjCMethod(method)
        self.instance_methods[objc_method.pyname] = objc_method
    libc.free(method_array)