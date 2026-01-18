import numpy as np
from pandas import (
import pandas._testing as tm
def test_iat_setitem_item_cache_cleared(indexer_ial, using_copy_on_write, warn_copy_on_write):
    data = {'x': np.arange(8, dtype=np.int64), 'y': np.int64(0)}
    df = DataFrame(data).copy()
    ser = df['y']
    with tm.assert_cow_warning(warn_copy_on_write):
        indexer_ial(df)[7, 0] = 9999
    with tm.assert_cow_warning(warn_copy_on_write):
        indexer_ial(df)[7, 1] = 1234
    assert df.iat[7, 1] == 1234
    if not using_copy_on_write:
        assert ser.iloc[-1] == 1234
    assert df.iloc[-1, -1] == 1234