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
def asized(self, *objs, **opts):
    """Size each object and return an **Asized** instance with
        size information and referents up to the given detail
        level (and with modified options, see method **set**).

        If only one object is given, the return value is the
        **Asized** instance for that object.  The **Asized** size
        of duplicate and ignored objects will be zero.
        """
    if opts:
        self.set(**opts)
    t = self._sizes(objs, Asized)
    return t[0] if len(t) == 1 else t