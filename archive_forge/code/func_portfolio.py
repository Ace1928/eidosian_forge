import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
@pytest.fixture
def portfolio(self):
    portfolio = DataFrame({'household_id': [1, 2, 2, 3, 3, 3, 4], 'asset_id': ['nl0000301109', 'nl0000289783', 'gb00b03mlx29', 'gb00b03mlx29', 'lu0197800237', 'nl0000289965', np.nan], 'name': ['ABN Amro', 'Robeco', 'Royal Dutch Shell', 'Royal Dutch Shell', 'AAB Eastern Europe Equity Fund', 'Postbank BioTech Fonds', np.nan], 'share': [1.0, 0.4, 0.6, 0.15, 0.6, 0.25, 1.0]}, columns=['household_id', 'asset_id', 'name', 'share']).set_index(['household_id', 'asset_id'])
    return portfolio