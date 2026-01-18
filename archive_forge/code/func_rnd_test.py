from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask import config
from dask.array.utils import assert_eq
def rnd_test(func, *args, **kwargs):
    a = func(*args, **kwargs)
    assert type(a._meta) == expect
    assert_eq(a, a)