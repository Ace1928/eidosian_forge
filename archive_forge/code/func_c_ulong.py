import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
def c_ulong(x):
    """Convert a Python integer to unsigned long"""
    return x