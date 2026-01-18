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
def assert_chunks_match(left, right):
    for ldim, rdim in zip(left, right):
        assert all((np.isnan(l) or l == r for l, r in zip(ldim, rdim)))