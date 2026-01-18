from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_divide_to_width():
    chunks = divide_to_width((8, 9, 10), 10)
    assert chunks == (8, 9, 10)
    chunks = divide_to_width((8, 2, 9, 10, 11, 12), 4)
    assert chunks == (4, 4, 2, 3, 3, 3, 3, 3, 4, 3, 4, 4, 4, 4, 4)