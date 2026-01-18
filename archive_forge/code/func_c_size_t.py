import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
def c_size_t(x):
    """Convert a Python integer to size_t"""
    return x