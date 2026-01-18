import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize(('rollings', 'key'), [({'on': 'a'}, 'a'), ({'on': None}, 'index')])
def test_groupby_rolling_nans_in_index(self, rollings, key):
    df = DataFrame({'a': to_datetime(['2020-06-01 12:00', '2020-06-01 14:00', np.nan]), 'b': [1, 2, 3], 'c': [1, 1, 1]})
    if key == 'index':
        df = df.set_index('a')
    with pytest.raises(ValueError, match=f'{key} values must not have NaT'):
        df.groupby('c').rolling('60min', **rollings)