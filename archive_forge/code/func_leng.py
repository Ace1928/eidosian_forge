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
def leng(obj, **opts):
    """Return the length of an object, in number of *items*.

    See function **basicsize** for a description of the available options.
    """
    n = t = _typedefof(obj, **opts)
    if t:
        n = t.leng
        if n and callable(n):
            i, v, n = (t.item, t.vari, n(obj))
            if v and i == _sizeof_Cbyte:
                i = getattr(obj, v, i)
                if i > _sizeof_Cbyte:
                    n = n // i
    return n