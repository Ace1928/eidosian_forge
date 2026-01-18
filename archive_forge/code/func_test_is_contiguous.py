from __future__ import annotations
import numpy as np
from numpy.testing import assert_array_equal
from xarray.core.nputils import NumpyVIndexAdapter, _is_contiguous
def test_is_contiguous() -> None:
    assert _is_contiguous([1])
    assert _is_contiguous([1, 2, 3])
    assert not _is_contiguous([1, 3])