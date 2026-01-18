from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_construct_ragged_array():
    rarray = RaggedArray([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]], dtype='int32')
    assert rarray.flat_array.dtype == 'int32'
    assert np.array_equal(rarray.flat_array, np.array([1, 2, 10, 20, 30, 11, 22, 33, 44], dtype='int32'))
    assert rarray.start_indices.dtype == 'uint8'
    assert np.array_equal(rarray.start_indices, np.array([0, 2, 2, 5, 5], dtype='uint64'))
    assert len(rarray) == 5
    assert rarray.isna().dtype == 'bool'
    assert np.array_equal(rarray.isna(), [False, True, False, True, False])
    expected = 9 * np.int32().nbytes + 5 * np.uint8().nbytes
    assert rarray.nbytes == expected
    assert type(rarray.dtype) == RaggedDtype