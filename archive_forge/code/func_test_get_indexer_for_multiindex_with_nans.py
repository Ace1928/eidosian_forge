from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_for_multiindex_with_nans(nulls_fixture):
    idx1 = MultiIndex.from_product([['A'], [1.0, 2.0]], names=['id1', 'id2'])
    idx2 = MultiIndex.from_product([['A'], [nulls_fixture, 2.0]], names=['id1', 'id2'])
    result = idx2.get_indexer(idx1)
    expected = np.array([-1, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    result = idx1.get_indexer(idx2)
    expected = np.array([-1, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)