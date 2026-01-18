from collections import namedtuple
from functools import singledispatch as simplegeneric
import importlib
import importlib.util
import importlib.machinery
import os
import os.path
import sys
from types import ModuleType
import warnings
def read_code(stream):
    import marshal
    magic = stream.read(4)
    if magic != importlib.util.MAGIC_NUMBER:
        return None
    stream.read(12)
    return marshal.load(stream)