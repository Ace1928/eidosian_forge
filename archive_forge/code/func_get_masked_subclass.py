import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def get_masked_subclass(*arrays):
    """
    Return the youngest subclass of MaskedArray from a list of (masked) arrays.

    In case of siblings, the first listed takes over.

    """
    if len(arrays) == 1:
        arr = arrays[0]
        if isinstance(arr, MaskedArray):
            rcls = type(arr)
        else:
            rcls = MaskedArray
    else:
        arrcls = [type(a) for a in arrays]
        rcls = arrcls[0]
        if not issubclass(rcls, MaskedArray):
            rcls = MaskedArray
        for cls in arrcls[1:]:
            if issubclass(cls, rcls):
                rcls = cls
    if rcls.__name__ == 'MaskedConstant':
        return MaskedArray
    return rcls