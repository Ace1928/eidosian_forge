from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def shared_range(flt_type, int_type):
    """Min and max in float type that are >=min, <=max in integer type

    This is not as easy as it sounds, because the float type may not be able to
    exactly represent the max or min integer values, so we have to find the
    next exactly representable floating point value to do the thresholding.

    Parameters
    ----------
    flt_type : dtype specifier
        A dtype specifier referring to a numpy floating point type.  For
        example, ``f4``, ``np.dtype('f4')``, ``np.float32`` are equivalent.
    int_type : dtype specifier
        A dtype specifier referring to a numpy integer type.  For example,
        ``i4``, ``np.dtype('i4')``, ``np.int32`` are equivalent

    Returns
    -------
    mn : object
        Number of type `flt_type` that is the minimum value in the range of
        `int_type`, such that ``mn.astype(int_type)`` >= min of `int_type`
    mx : object
        Number of type `flt_type` that is the maximum value in the range of
        `int_type`, such that ``mx.astype(int_type)`` <= max of `int_type`

    Examples
    --------
    >>> shared_range(np.float32, np.int32) == (-2147483648.0, 2147483520.0)
    True
    >>> shared_range('f4', 'i4') == (-2147483648.0, 2147483520.0)
    True
    """
    flt_type = np.dtype(flt_type).type
    int_type = np.dtype(int_type).type
    key = (flt_type, int_type)
    try:
        return _SHARED_RANGES[key]
    except KeyError:
        pass
    ii = np.iinfo(int_type)
    fi = np.finfo(flt_type)
    mn = ceil_exact(ii.min, flt_type)
    if mn == -np.inf:
        mn = fi.min
    mx = floor_exact(ii.max, flt_type)
    if mx == np.inf:
        mx = fi.max
    elif TRUNC_UINT64 and int_type == np.uint64:
        mx = min(mx, flt_type(2 ** 63))
    _SHARED_RANGES[key] = (mn, mx)
    return (mn, mx)