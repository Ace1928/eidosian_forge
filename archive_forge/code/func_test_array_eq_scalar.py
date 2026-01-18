from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
@pytest.mark.parametrize('scalar', [np.array([1, 2]), [1, 2]])
def test_array_eq_scalar(scalar):
    arg1 = [[1, 2], [], [1, 2], [1, 3], [11, 22, 33, 44]]
    ra = RaggedArray(arg1, dtype='int32')
    result = ra == scalar
    expected = np.array([1, 0, 1, 0, 0], dtype='bool')
    np.testing.assert_array_equal(result, expected)
    result_negated = ra != scalar
    expected_negated = ~expected
    np.testing.assert_array_equal(result_negated, expected_negated)