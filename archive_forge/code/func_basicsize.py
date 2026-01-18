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
def basicsize(obj, **opts):
    """Return the basic size of an object (in bytes).

    The available options and defaults are:

        *derive=False* -- derive type from super type

        *infer=False*  -- try to infer types

        *save=False*   -- save the object's type definition if new

    See this module documentation for the definition of *basic size*.
    """
    b = t = _typedefof(obj, **opts)
    if t:
        b = t.base
    return b