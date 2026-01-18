import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
def c_ubyte(c):
    if not 0 <= c < 256:
        raise OverflowError()
    return ctypes.c_ubyte(c)