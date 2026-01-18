import types
import sys
import numbers
import functools
import copy
import inspect
def bytes_to_native_str(b, encoding=None):
    return native(b)