from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def floor_log2(x):
    """floor of log2 of abs(`x`)

    Embarrassingly, from https://en.wikipedia.org/wiki/Binary_logarithm

    Parameters
    ----------
    x : int

    Returns
    -------
    L : None or int
        floor of base 2 log of `x`.  None if `x` == 0.

    Examples
    --------
    >>> floor_log2(2**9+1)
    9
    >>> floor_log2(-2**9+1)
    8
    >>> floor_log2(0.5)
    -1
    >>> floor_log2(0) is None
    True
    """
    ip = 0
    rem = abs(x)
    if rem > 1:
        while rem >= 2:
            ip += 1
            rem //= 2
        return ip
    elif rem == 0:
        return None
    while rem < 1:
        ip -= 1
        rem *= 2
    return ip