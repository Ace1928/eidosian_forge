import os
from ctypes.macholib.framework import framework_info
from ctypes.macholib.dylib import dylib_info
from itertools import *
def dyld_find(name, executable_path=None, env=None):
    """
    Find a library or framework using dyld semantics
    """
    for path in dyld_image_suffix_search(chain(dyld_override_search(name, env), dyld_executable_path_search(name, executable_path), dyld_default_search(name, env)), env):
        if os.path.isfile(path):
            return path
        try:
            if _dyld_shared_cache_contains_path(path):
                return path
        except NotImplementedError:
            pass
    raise ValueError('dylib %s could not be found' % (name,))