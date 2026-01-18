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
def exclude_objs(self, *objs):
    """Exclude the specified objects from sizing, profiling and ranking."""
    for o in objs:
        self._seen.setdefault(id(o), -1)