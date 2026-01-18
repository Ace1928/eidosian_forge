from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_get_item_scalar():
    arg = [[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]]
    rarray = RaggedArray(arg, dtype='float16')
    for i, expected in enumerate(arg):
        result = rarray[i]
        if expected is None:
            expected = np.array([], dtype='float16')
        if isinstance(result, np.ndarray):
            assert result.dtype == 'float16'
        else:
            assert np.isnan(result)
        np.testing.assert_array_equal(result, expected)
    for i, expected in enumerate(arg):
        result = rarray[i - 5]
        if expected is None:
            expected = np.array([], dtype='float16')
        if isinstance(result, np.ndarray):
            assert result.dtype == 'float16'
        else:
            assert np.isnan(result)
        np.testing.assert_array_equal(result, expected)