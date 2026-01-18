import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('indexer, pos', [([], []), (['A'], slice(3)), (['A', 'D'], []), (['D', 'E'], []), (['D'], []), (pd.IndexSlice[:, ['foo']], slice(2, None, 3)), (pd.IndexSlice[:, ['foo', 'bah']], slice(2, None, 3))])
def test_loc_getitem_duplicates_multiindex_missing_indexers(indexer, pos):
    idx = MultiIndex.from_product([['A', 'B', 'C'], ['foo', 'bar', 'baz']], names=['one', 'two'])
    ser = Series(np.arange(9, dtype='int64'), index=idx).sort_index()
    expected = ser.iloc[pos]
    if expected.size == 0 and indexer != []:
        with pytest.raises(KeyError, match=str(indexer)):
            ser.loc[indexer]
    elif indexer == (slice(None), ['foo', 'bah']):
        with pytest.raises(KeyError, match="'bah'"):
            ser.loc[indexer]
    else:
        result = ser.loc[indexer]
        tm.assert_series_equal(result, expected)