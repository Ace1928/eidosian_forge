import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_concat_bad_index_levels(styler, df):
    df = df.copy()
    df.index = MultiIndex.from_tuples([(0, 0), (1, 1)])
    msg = 'number of index levels must be same in `other`'
    with pytest.raises(ValueError, match=msg):
        styler.concat(df.style)