from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.reshape import contract_tuple, expand_tuple, reshape_rechunk
from dask.array.utils import assert_eq
def test_expand_tuple():
    assert expand_tuple((2, 4), 2) == (1, 1, 2, 2)
    assert expand_tuple((2, 4), 3) == (1, 1, 1, 1, 2)
    assert expand_tuple((3, 4), 2) == (1, 2, 2, 2)
    assert expand_tuple((7, 4), 3) == (2, 2, 3, 1, 1, 2)