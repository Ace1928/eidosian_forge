from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.utils import assert_eq
def test_min_max_round_funcs():
    image = da.from_array(np.array([[0, 1], [1, 2]]), chunks=(1, 2))
    assert int(np.min(image)) == 0
    assert int(np.max(image)) == 2
    assert np.round(image)[1, 1] == 2