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
def default_fill_value(obj):
    """
    Return the default fill value for the argument object.

    The default filling value depends on the datatype of the input
    array or the type of the input scalar:

       ========  ========
       datatype  default
       ========  ========
       bool      True
       int       999999
       float     1.e20
       complex   1.e20+0j
       object    '?'
       string    'N/A'
       ========  ========

    For structured types, a structured scalar is returned, with each field the
    default fill value for its type.

    For subarray types, the fill value is an array of the same size containing
    the default scalar fill value.

    Parameters
    ----------
    obj : ndarray, dtype or scalar
        The array data-type or scalar for which the default fill value
        is returned.

    Returns
    -------
    fill_value : scalar
        The default fill value.

    Examples
    --------
    >>> np.ma.default_fill_value(1)
    999999
    >>> np.ma.default_fill_value(np.array([1.1, 2., np.pi]))
    1e+20
    >>> np.ma.default_fill_value(np.dtype(complex))
    (1e+20+0j)

    """

    def _scalar_fill_value(dtype):
        if dtype.kind in 'Mm':
            return default_filler.get(dtype.str[1:], '?')
        else:
            return default_filler.get(dtype.kind, '?')
    dtype = _get_dtype_of(obj)
    return _recursive_fill_value(dtype, _scalar_fill_value)