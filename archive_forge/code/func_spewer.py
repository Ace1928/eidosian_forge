from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def spewer(frame, s, ignored):
    """
    A trace function for sys.settrace that prints every function or method call.
    """
    from twisted.python import reflect
    if 'self' in frame.f_locals:
        se = frame.f_locals['self']
        if hasattr(se, '__class__'):
            k = reflect.qual(se.__class__)
        else:
            k = reflect.qual(type(se))
        print(f'method {frame.f_code.co_name} of {k} at {id(se)}')
    else:
        print('function %s in %s, line %s' % (frame.f_code.co_name, frame.f_code.co_filename, frame.f_lineno))