from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_array_eq_ragged():
    arg1 = [[1, 2], [], [1, 2], [3, 2, 1], [11, 22, 33, 44]]
    ra1 = RaggedArray(arg1, dtype='int32')
    arg2 = [[1, 2], [2, 3, 4, 5], [1, 2], [11, 22, 33], [11]]
    ra2 = RaggedArray(arg2, dtype='int32')
    result = ra1 == ra2
    expected = np.array([1, 0, 1, 0, 0], dtype='bool')
    np.testing.assert_array_equal(result, expected)
    result_negated = ra1 != ra2
    expected_negated = ~expected
    np.testing.assert_array_equal(result_negated, expected_negated)