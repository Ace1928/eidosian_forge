import os
from ctypes.macholib.framework import framework_info
from ctypes.macholib.dylib import dylib_info
from itertools import *
def dyld_library_path(env=None):
    return dyld_env(env, 'DYLD_LIBRARY_PATH')