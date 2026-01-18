import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.indexers import check_array_indexer
@pytest.mark.parametrize('indexer', [[0, 1, None], pd.array([0, 1, pd.NA], dtype='Int64')])
def test_int_raise_missing_values(indexer):
    arr = np.array([1, 2, 3])
    msg = 'Cannot index with an integer indexer containing NA values'
    with pytest.raises(ValueError, match=msg):
        check_array_indexer(arr, indexer)