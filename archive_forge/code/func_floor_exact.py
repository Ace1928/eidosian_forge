from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def floor_exact(val, flt_type):
    """Return nearest exact integer <= `val` in float type `flt_type`

    Parameters
    ----------
    val : int
        We have to pass val as an int rather than the floating point type
        because large integers cast as floating point may be rounded by the
        casting process.
    flt_type : numpy type
        numpy float type.

    Returns
    -------
    floor_val : object
        value of same floating point type as `val`, that is the nearest exact
        integer in this type such that `floor_val` <= `val`.  Thus if `val` is
        exact in `flt_type`, `floor_val` == `val`.

    Examples
    --------
    Obviously 2 is within the range of representable integers for float32

    >>> floor_exact(2, np.float32)
    2.0

    As is 2**24-1 (the number of significand digits is 23 + 1 implicit)

    >>> floor_exact(2**24-1, np.float32) == 2**24-1
    True

    But 2**24+1 gives a number that float32 can't represent exactly

    >>> floor_exact(2**24+1, np.float32) == 2**24
    True

    As for the numpy floor function, negatives floor towards -inf

    >>> floor_exact(-2**24-1, np.float32) == -2**24-2
    True
    """
    val = int(val)
    flt_type = np.dtype(flt_type).type
    sign = 1 if val > 0 else -1
    try:
        fval = flt_type(val)
    except OverflowError:
        return sign * np.inf
    if not np.isfinite(fval):
        return fval
    info = type_info(flt_type)
    diff = val - int(fval)
    if diff >= 0:
        return fval
    biggest_gap = 2 ** (floor_log2(val) - info['nmant'])
    assert biggest_gap > 1
    fval -= flt_type(biggest_gap)
    return fval