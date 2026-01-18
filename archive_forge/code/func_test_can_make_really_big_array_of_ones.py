from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq
from dask.array.wrap import ones
def test_can_make_really_big_array_of_ones():
    ones((1000000, 1000000), chunks=(100000, 100000))
    ones(shape=(1000000, 1000000), chunks=(100000, 100000))