import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_get_indexer_arrow_dictionary_target(self):
    pa = pytest.importorskip('pyarrow')
    target = Index(ArrowExtensionArray(pa.array([1, 2], type=pa.dictionary(pa.int8(), pa.int8()))))
    idx = Index([1])
    result = idx.get_indexer(target)
    expected = np.array([0, -1], dtype=np.int64)
    tm.assert_numpy_array_equal(result, expected)
    result_1, result_2 = idx.get_indexer_non_unique(target)
    expected_1, expected_2 = (np.array([0, -1], dtype=np.int64), np.array([1], dtype=np.int64))
    tm.assert_numpy_array_equal(result_1, expected_1)
    tm.assert_numpy_array_equal(result_2, expected_2)