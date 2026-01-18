import sys
import os
import os.path
import ctypes
from ctypes import c_char_p, c_int, c_size_t, c_void_p, pointer, CFUNCTYPE, POINTER
import ctypes.util
import platform
import textwrap
def dict_remove_from_session(dict, word):
    return dict_remove_from_session1(dict, word, len(word))