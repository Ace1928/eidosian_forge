import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_datelike_on_not_monotonic_within_each_group(self):
    df = DataFrame({'A': [1] * 3 + [2] * 3, 'B': [Timestamp(year, 1, 1) for year in [2020, 2021, 2019]] * 2, 'C': range(6)})
    with pytest.raises(ValueError, match='Each group within B must be monotonic.'):
        df.groupby('A').rolling('365D', on='B')