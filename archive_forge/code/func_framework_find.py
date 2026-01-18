import os
from ctypes.macholib.framework import framework_info
from ctypes.macholib.dylib import dylib_info
from itertools import *
def framework_find(fn, executable_path=None, env=None):
    """
    Find a framework using dyld semantics in a very loose manner.

    Will take input such as:
        Python
        Python.framework
        Python.framework/Versions/Current
    """
    error = None
    try:
        return dyld_find(fn, executable_path=executable_path, env=env)
    except ValueError as e:
        error = e
    fmwk_index = fn.rfind('.framework')
    if fmwk_index == -1:
        fmwk_index = len(fn)
        fn += '.framework'
    fn = os.path.join(fn, os.path.basename(fn[:fmwk_index]))
    try:
        return dyld_find(fn, executable_path=executable_path, env=env)
    except ValueError:
        raise error
    finally:
        error = None