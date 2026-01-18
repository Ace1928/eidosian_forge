from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def mystats(x, y):
    return np.std(x, axis=-1) * np.mean(y, axis=-1)