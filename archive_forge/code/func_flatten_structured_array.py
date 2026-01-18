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
def flatten_structured_array(a):
    """
    Flatten a structured array.

    The data type of the output is chosen such that it can represent all of the
    (nested) fields.

    Parameters
    ----------
    a : structured array

    Returns
    -------
    output : masked array or ndarray
        A flattened masked array if the input is a masked array, otherwise a
        standard ndarray.

    Examples
    --------
    >>> ndtype = [('a', int), ('b', float)]
    >>> a = np.array([(1, 1), (2, 2)], dtype=ndtype)
    >>> np.ma.flatten_structured_array(a)
    array([[1., 1.],
           [2., 2.]])

    """

    def flatten_sequence(iterable):
        """
        Flattens a compound of nested iterables.

        """
        for elm in iter(iterable):
            if hasattr(elm, '__iter__'):
                yield from flatten_sequence(elm)
            else:
                yield elm
    a = np.asanyarray(a)
    inishape = a.shape
    a = a.ravel()
    if isinstance(a, MaskedArray):
        out = np.array([tuple(flatten_sequence(d.item())) for d in a._data])
        out = out.view(MaskedArray)
        out._mask = np.array([tuple(flatten_sequence(d.item())) for d in getmaskarray(a)])
    else:
        out = np.array([tuple(flatten_sequence(d.item())) for d in a])
    if len(inishape) > 1:
        newshape = list(out.shape)
        newshape[0] = inishape
        out.shape = tuple(flatten_sequence(newshape))
    return out