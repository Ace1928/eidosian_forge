import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_transform(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    s = frame['A']
    grouper = s.index.get_level_values(0)
    grouped = s.groupby(grouper, group_keys=False)
    applied = grouped.apply(lambda x: x * 2)
    expected = grouped.transform(lambda x: x * 2)
    result = applied.reindex(expected.index)
    tm.assert_series_equal(result, expected, check_names=False)