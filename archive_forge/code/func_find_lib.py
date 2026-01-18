import os
import sys
import unittest
from ctypes.macholib.dyld import dyld_find
from ctypes.macholib.dylib import dylib_info
from ctypes.macholib.framework import framework_info
def find_lib(name):
    possible = ['lib' + name + '.dylib', name + '.dylib', name + '.framework/' + name]
    for dylib in possible:
        try:
            return os.path.realpath(dyld_find(dylib))
        except ValueError:
            pass
    raise ValueError('%s not found' % (name,))