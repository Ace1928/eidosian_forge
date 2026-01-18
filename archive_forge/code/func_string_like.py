from typing import Any, Optional
from collections.abc import Mapping
import numpy as np
import pandas as pd
def string_like(value, name, optional=False, options=None, lower=True):
    """
    Check if object is string-like and raise if not

    Parameters
    ----------
    value : object
        Value to verify.
    name : str
        Variable name for exceptions.
    optional : bool
        Flag indicating whether None is allowed.
    options : tuple[str]
        Allowed values for input parameter `value`.
    lower : bool
        Convert all case-based characters in `value` into lowercase.

    Returns
    -------
    str
        The validated input

    Raises
    ------
    TypeError
        If the value is not a string or None when optional is True.
    ValueError
        If the input is not in ``options`` when ``options`` is set.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        extra_text = ' or None' if optional else ''
        raise TypeError(f'{name} must be a string{extra_text}')
    if lower:
        value = value.lower()
    if options is not None and value not in options:
        extra_text = 'If not None, ' if optional else ''
        options_text = "'" + "', '".join(options) + "'"
        msg = '{}{} must be one of: {}'.format(extra_text, name, options_text)
        raise ValueError(msg)
    return value