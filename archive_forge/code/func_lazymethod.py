import errno
import functools
import os
import io
import pickle
import sys
import time
import string
import warnings
from importlib import import_module
from math import sin, cos, radians, atan2, degrees
from contextlib import contextmanager, ExitStack
from math import gcd
from pathlib import PurePath, Path
import re
import numpy as np
from ase.formula import formula_hill, formula_metal
def lazymethod(meth):
    """Decorator for lazy evaluation and caching of data.

    Example::

      class MyClass:

         @lazymethod
         def thing(self):
             return expensive_calculation()

    The method body is only executed first time thing() is called, and
    its return value is stored.  Subsequent calls return the cached
    value."""
    name = meth.__name__

    @functools.wraps(meth)
    def getter(self):
        try:
            cache = self._lazy_cache
        except AttributeError:
            cache = self._lazy_cache = {}
        if name not in cache:
            cache[name] = meth(self)
        return cache[name]
    return getter