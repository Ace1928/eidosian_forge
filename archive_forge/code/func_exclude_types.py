import sys
import types as Types
import warnings
import weakref as Weakref
from inspect import isbuiltin, isclass, iscode, isframe, isfunction, ismethod, ismodule
from math import log
from os import curdir, linesep
from struct import calcsize
from gc import get_objects as _getobjects
from gc import get_referents as _getreferents  # containers only?
from array import array as _array  # array type
def exclude_types(self, *objs):
    """Exclude the specified object instances and types from sizing.

        All instances and types of the given objects are excluded,
        even objects specified as positional arguments in subsequent
        calls to methods **asizeof** and **asizesof**.
        """
    for o in objs:
        for t in _key2tuple(o):
            if t and t not in self._excl_d:
                self._excl_d[t] = 0