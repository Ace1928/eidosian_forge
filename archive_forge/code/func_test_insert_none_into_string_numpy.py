import numpy as np
import pytest
from pandas import Index
import pandas._testing as tm
def test_insert_none_into_string_numpy(self):
    pytest.importorskip('pyarrow')
    index = Index(['a', 'b', 'c'], dtype='string[pyarrow_numpy]')
    result = index.insert(-1, None)
    expected = Index(['a', 'b', None, 'c'], dtype='string[pyarrow_numpy]')
    tm.assert_index_equal(result, expected)