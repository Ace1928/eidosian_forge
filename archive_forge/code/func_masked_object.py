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
def masked_object(x, value, copy=True, shrink=True):
    """
    Mask the array `x` where the data are exactly equal to value.

    This function is similar to `masked_values`, but only suitable
    for object arrays: for floating point, use `masked_values` instead.

    Parameters
    ----------
    x : array_like
        Array to mask
    value : object
        Comparison value
    copy : {True, False}, optional
        Whether to return a copy of `x`.
    shrink : {True, False}, optional
        Whether to collapse a mask full of False to nomask

    Returns
    -------
    result : MaskedArray
        The result of masking `x` where equal to `value`.

    See Also
    --------
    masked_where : Mask where a condition is met.
    masked_equal : Mask where equal to a given value (integers).
    masked_values : Mask using floating point equality.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> food = np.array(['green_eggs', 'ham'], dtype=object)
    >>> # don't eat spoiled food
    >>> eat = ma.masked_object(food, 'green_eggs')
    >>> eat
    masked_array(data=[--, 'ham'],
                 mask=[ True, False],
           fill_value='green_eggs',
                dtype=object)
    >>> # plain ol` ham is boring
    >>> fresh_food = np.array(['cheese', 'ham', 'pineapple'], dtype=object)
    >>> eat = ma.masked_object(fresh_food, 'green_eggs')
    >>> eat
    masked_array(data=['cheese', 'ham', 'pineapple'],
                 mask=False,
           fill_value='green_eggs',
                dtype=object)

    Note that `mask` is set to ``nomask`` if possible.

    >>> eat
    masked_array(data=['cheese', 'ham', 'pineapple'],
                 mask=False,
           fill_value='green_eggs',
                dtype=object)

    """
    if isMaskedArray(x):
        condition = umath.equal(x._data, value)
        mask = x._mask
    else:
        condition = umath.equal(np.asarray(x), value)
        mask = nomask
    mask = mask_or(mask, make_mask(condition, shrink=shrink))
    return masked_array(x, mask=mask, copy=copy, fill_value=value)