from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def myfilter(x, cn=10, axis=-1):
    y = np.fft.fft(x, axis=axis)
    y[cn:-cn] = 0
    nx = np.fft.ifft(y, axis=axis)
    return np.real(nx)