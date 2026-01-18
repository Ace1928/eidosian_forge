from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.fixture
def nulls_df():
    n = np.nan
    return DataFrame({'A': [1, 1, n, 4, n, 6, 6, 6, 6], 'B': [1, 1, 3, n, n, 6, 6, 6, 6], 'C': [1, 2, 3, 4, 5, 6, n, 8, n], 'D': [1, 2, 3, 4, 5, 6, 7, n, n]})