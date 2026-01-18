import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def get_instance_method(self, name):
    """Returns a python representation of the named instance method,
        either by looking it up in the cached list of methods or by searching
        for and creating a new method object."""
    if name in self.instance_methods:
        return self.instance_methods[name]
    else:
        selector = get_selector(name.replace(b'_', b':'))
        method = c_void_p(objc.class_getInstanceMethod(self.ptr, selector))
        if method.value:
            objc_method = ObjCMethod(method)
            self.instance_methods[name] = objc_method
            return objc_method
    return None