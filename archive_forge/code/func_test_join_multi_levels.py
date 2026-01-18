import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_join_multi_levels(self, portfolio, household, expected):
    portfolio = portfolio.copy()
    household = household.copy()
    result = household.join(portfolio, how='inner')
    tm.assert_frame_equal(result, expected)