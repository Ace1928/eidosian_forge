import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_join_multi_levels_invalid(self, portfolio, household):
    portfolio = portfolio.copy()
    household = household.copy()
    household.index.name = 'foo'
    with pytest.raises(ValueError, match='cannot join with no overlapping index names'):
        household.join(portfolio, how='inner')
    portfolio2 = portfolio.copy()
    portfolio2.index.set_names(['household_id', 'foo'])
    with pytest.raises(ValueError, match='columns overlap but no suffix specified'):
        portfolio2.join(portfolio, how='inner')