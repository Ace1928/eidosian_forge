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
def flatsize(obj, align=0, **opts):
    """Return the flat size of an object (in bytes), optionally aligned
    to the given power-of-2.

    See function **basicsize** for a description of other available options.

    See this module documentation for the definition of *flat size*.
    """
    f = t = _typedefof(obj, **opts)
    if t:
        if align > 1:
            m = align - 1
            if m & align:
                raise _OptionError(flatsize, align=align)
        else:
            m = 0
        f = t.flat(obj, mask=m)
    return f