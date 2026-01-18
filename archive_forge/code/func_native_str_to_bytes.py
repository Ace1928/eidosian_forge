import types
import sys
import numbers
import functools
import copy
import inspect
def native_str_to_bytes(s, encoding=None):
    from future.types import newbytes
    return newbytes(s)