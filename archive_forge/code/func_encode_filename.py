import types
import sys
import numbers
import functools
import copy
import inspect
def encode_filename(filename):
    if PY3:
        return filename
    else:
        if isinstance(filename, unicode):
            return filename.encode('utf-8')
        return filename