import sys
import os
import os.path
import ctypes
from ctypes import c_char_p, c_int, c_size_t, c_void_p, pointer, CFUNCTYPE, POINTER
import ctypes.util
import platform
import textwrap
def dict_store_replacement(dict, mis, cor):
    return dict_store_replacement1(dict, mis, len(mis), cor, len(cor))