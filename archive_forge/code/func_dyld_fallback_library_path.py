import os
from ctypes.macholib.framework import framework_info
from ctypes.macholib.dylib import dylib_info
from itertools import *
def dyld_fallback_library_path(env=None):
    return dyld_env(env, 'DYLD_FALLBACK_LIBRARY_PATH')