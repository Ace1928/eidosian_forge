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
def asizesof(self, *objs, **opts):
    """Return the individual sizes of the given objects
        (with modified options, see method  **set**).

        The size of duplicate and ignored objects will be zero.
        """
    if opts:
        self.set(**opts)
    return self._sizes(objs, None)