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
def mergeFunctionMetadata(f, g):
    """
    Overwrite C{g}'s name and docstring with values from C{f}.  Update
    C{g}'s instance dictionary with C{f}'s.

    @return: A function that has C{g}'s behavior and metadata merged from
        C{f}.
    """
    try:
        g.__name__ = f.__name__
    except TypeError:
        pass
    try:
        g.__doc__ = f.__doc__
    except (TypeError, AttributeError):
        pass
    try:
        g.__dict__.update(f.__dict__)
    except (TypeError, AttributeError):
        pass
    try:
        g.__module__ = f.__module__
    except TypeError:
        pass
    return g