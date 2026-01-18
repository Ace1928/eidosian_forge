import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage
import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray
import unittest
def nulp_diff(x, y, dtype=None):
    """For each item in x and y, return the number of representable floating
    points between them.

    Parameters
    ----------
    x : array_like
        first input array
    y : array_like
        second input array
    dtype : dtype, optional
        Data-type to convert `x` and `y` to if given. Default is None.

    Returns
    -------
    nulp : array_like
        number of representable floating point numbers between each item in x
        and y.

    Notes
    -----
    For computing the ULP difference, this API does not differentiate between
    various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
    is zero).

    Examples
    --------
    # By definition, epsilon is the smallest number such as 1 + eps != 1, so
    # there should be exactly one ULP between 1 and 1 + eps
    >>> nulp_diff(1, 1 + np.finfo(x.dtype).eps)  # doctest: +SKIP
    1.0
    """
    import numpy as np
    if dtype:
        x = np.asarray(x, dtype=dtype)
        y = np.asarray(y, dtype=dtype)
    else:
        x = np.asarray(x)
        y = np.asarray(y)
    t = np.common_type(x, y)
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise NotImplementedError('_nulp not implemented for complex array')
    x = np.array([x], dtype=t)
    y = np.array([y], dtype=t)
    x[np.isnan(x)] = np.nan
    y[np.isnan(y)] = np.nan
    if not x.shape == y.shape:
        raise ValueError(f'x and y do not have the same shape: {x.shape} - {y.shape}')

    def _diff(rx, ry, vdt):
        diff = np.asarray(rx - ry, dtype=vdt)
        return np.abs(diff)
    rx = integer_repr(x)
    ry = integer_repr(y)
    return _diff(rx, ry, t)