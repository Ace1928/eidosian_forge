from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def best_float():
    """Floating point type with best precision

    This is nearly always np.longdouble, except on Windows, where np.longdouble
    is Intel80 storage, but with float64 precision for calculations.  In that
    case we return float64 on the basis it's the fastest and smallest at the
    highest precision.

    SPARC float128 also proved so slow that we prefer float64.

    Returns
    -------
    best_type : numpy type
        floating point type with highest precision

    Notes
    -----
    Needs to run without error for module import, because it is called in
    ``ok_floats`` below, and therefore in setting module global ``OK_FLOATS``.
    """
    try:
        long_info = type_info(np.longdouble)
    except FloatingError:
        return np.float64
    if long_info['nmant'] > type_info(np.float64)['nmant'] and machine() != 'sparc64':
        return np.longdouble
    return np.float64