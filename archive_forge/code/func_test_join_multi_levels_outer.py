import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_join_multi_levels_outer(self, portfolio, household, expected):
    portfolio = portfolio.copy()
    household = household.copy()
    result = household.join(portfolio, how='outer')
    expected = concat([expected, DataFrame({'share': [1.0]}, index=MultiIndex.from_tuples([(4, np.nan)], names=['household_id', 'asset_id']))], axis=0, sort=True).reindex(columns=expected.columns)
    tm.assert_frame_equal(result, expected, check_index_type=False)