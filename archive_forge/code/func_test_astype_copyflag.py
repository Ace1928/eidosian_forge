import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
def test_astype_copyflag():
    arr = np.arange(10, dtype=np.intp)
    res_true = arr.astype(np.intp, copy=True)
    assert not np.may_share_memory(arr, res_true)
    res_always = arr.astype(np.intp, copy=np._CopyMode.ALWAYS)
    assert not np.may_share_memory(arr, res_always)
    res_false = arr.astype(np.intp, copy=False)
    assert np.may_share_memory(arr, res_false)
    res_if_needed = arr.astype(np.intp, copy=np._CopyMode.IF_NEEDED)
    assert np.may_share_memory(arr, res_if_needed)
    res_never = arr.astype(np.intp, copy=np._CopyMode.NEVER)
    assert np.may_share_memory(arr, res_never)
    res_false = arr.astype(np.float64, copy=False)
    assert_array_equal(res_false, arr)
    res_if_needed = arr.astype(np.float64, copy=np._CopyMode.IF_NEEDED)
    assert_array_equal(res_if_needed, arr)
    assert_raises(ValueError, arr.astype, np.float64, copy=np._CopyMode.NEVER)