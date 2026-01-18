import os
import re
import sys
import ctypes
import ctypes.util
import pyglet
@staticmethod
def load_framework(name):
    path = ctypes.util.find_library(name)
    if path is None:
        frameworks = {'AGL': '/System/Library/Frameworks/AGL.framework/AGL', 'IOKit': '/System/Library/Frameworks/IOKit.framework/IOKit', 'OpenAL': '/System/Library/Frameworks/OpenAL.framework/OpenAL', 'OpenGL': '/System/Library/Frameworks/OpenGL.framework/OpenGL'}
        path = frameworks.get(name)
    if path:
        lib = ctypes.cdll.LoadLibrary(path)
        if _debug_lib:
            print(path)
        if _debug_trace:
            lib = _TraceLibrary(lib)
        return lib
    raise ImportError(f"Can't find framework {name}.")