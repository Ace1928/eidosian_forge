from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def longdouble_precision_improved():
    """True if longdouble precision increased since initial import

    This can happen on Windows compiled with MSVC.  It may be because libraries
    compiled with mingw (longdouble is Intel80) get linked to numpy compiled
    with MSVC (longdouble is Float64)
    """
    return not longdouble_lte_float64() and _LD_LTE_FLOAT64