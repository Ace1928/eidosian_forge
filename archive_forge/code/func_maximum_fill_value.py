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
def maximum_fill_value(obj):
    """
    Return the minimum value that can be represented by the dtype of an object.

    This function is useful for calculating a fill value suitable for
    taking the maximum of an array with a given dtype.

    Parameters
    ----------
    obj : ndarray, dtype or scalar
        An object that can be queried for it's numeric type.

    Returns
    -------
    val : scalar
        The minimum representable value.

    Raises
    ------
    TypeError
        If `obj` isn't a suitable numeric type.

    See Also
    --------
    minimum_fill_value : The inverse function.
    set_fill_value : Set the filling value of a masked array.
    MaskedArray.fill_value : Return current fill value.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.int8()
    >>> ma.maximum_fill_value(a)
    -128
    >>> a = np.int32()
    >>> ma.maximum_fill_value(a)
    -2147483648

    An array of numeric data can also be passed.

    >>> a = np.array([1, 2, 3], dtype=np.int8)
    >>> ma.maximum_fill_value(a)
    -128
    >>> a = np.array([1, 2, 3], dtype=np.float32)
    >>> ma.maximum_fill_value(a)
    -inf

    """
    return _extremum_fill_value(obj, max_filler, 'maximum')