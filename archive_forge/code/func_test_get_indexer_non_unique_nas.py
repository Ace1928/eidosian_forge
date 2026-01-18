from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.missing import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Index
import pandas._testing as tm
def test_get_indexer_non_unique_nas(self, nulls_fixture, request, using_infer_string):
    if using_infer_string and (nulls_fixture is None or nulls_fixture is NA):
        request.applymarker(pytest.mark.xfail(reason='NAs are cast to NaN'))
    index = Index(['a', 'b', nulls_fixture])
    indexer, missing = index.get_indexer_non_unique([nulls_fixture])
    expected_indexer = np.array([2], dtype=np.intp)
    expected_missing = np.array([], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected_indexer)
    tm.assert_numpy_array_equal(missing, expected_missing)
    index = Index(['a', nulls_fixture, 'b', nulls_fixture])
    indexer, missing = index.get_indexer_non_unique([nulls_fixture])
    expected_indexer = np.array([1, 3], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected_indexer)
    tm.assert_numpy_array_equal(missing, expected_missing)
    if is_matching_na(nulls_fixture, float('NaN')):
        index = Index(['a', float('NaN'), 'b', float('NaN')])
        match_but_not_identical = True
    elif is_matching_na(nulls_fixture, Decimal('NaN')):
        index = Index(['a', Decimal('NaN'), 'b', Decimal('NaN')])
        match_but_not_identical = True
    else:
        match_but_not_identical = False
    if match_but_not_identical:
        indexer, missing = index.get_indexer_non_unique([nulls_fixture])
        expected_indexer = np.array([1, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)