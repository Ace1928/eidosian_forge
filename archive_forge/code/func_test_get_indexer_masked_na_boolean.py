import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('dtype', ['boolean', 'bool[pyarrow]'])
def test_get_indexer_masked_na_boolean(self, dtype):
    if dtype == 'bool[pyarrow]':
        pytest.importorskip('pyarrow')
    idx = Index([True, False, NA], dtype=dtype)
    result = idx.get_loc(False)
    assert result == 1
    result = idx.get_loc(NA)
    assert result == 2