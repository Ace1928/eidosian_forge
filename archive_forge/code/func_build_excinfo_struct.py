from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def build_excinfo_struct(self, exc, exc_args, loc, func_name):
    if loc is not None:
        fname = loc._raw_function_name()
        if fname is None:
            fname = func_name
        locinfo = (fname, loc.filename, loc.line)
        if None in locinfo:
            locinfo = None
    else:
        locinfo = None
    exc = (exc, exc_args, locinfo)
    return exc