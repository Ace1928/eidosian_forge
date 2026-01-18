import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.xfail(reason='Tries label-based and raises KeyError; in some cases raises when calling np.asarray')
@pytest.mark.parametrize('idx', [[0, 1, 2, pd.NA], pd.array([0, 1, 2, pd.NA], dtype='Int64')], ids=['list', 'integer-array'])
def test_getitem_series_integer_with_missing_raises(self, data, idx):
    msg = 'Cannot index with an integer indexer containing NA values'
    ser = pd.Series(data, index=[chr(100 + i) for i in range(len(data))])
    with pytest.raises(ValueError, match=msg):
        ser[idx]