from typing import Any, Optional
from collections.abc import Mapping
import numpy as np
import pandas as pd
def float_like(value, name, optional=False, strict=False):
    """
    Convert to float or raise if not float_like

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow int, np.integer, float or np.inexact that are
        not bool or complex. If False, allow complex types with 0 imag part or
        any other type that is float like in the sense that it support
        multiplication by 1.0 and conversion to float.

    Returns
    -------
    converted : float
        value converted to a float
    """
    if optional and value is None:
        return None
    is_bool = isinstance(value, bool)
    is_complex = isinstance(value, (complex, np.complexfloating))
    if hasattr(value, 'squeeze') and callable(value.squeeze):
        value = value.squeeze()
    if isinstance(value, (int, np.integer, float, np.inexact)) and (not (is_bool or is_complex)):
        return float(value)
    elif not strict and is_complex:
        imag = np.imag(value)
        if imag == 0:
            return float(np.real(value))
    elif not strict and (not is_bool):
        try:
            return float(value / 1.0)
        except Exception:
            pass
    extra_text = ' or None' if optional else ''
    raise TypeError('{} must be float_like (float or np.inexact){}'.format(name, extra_text))