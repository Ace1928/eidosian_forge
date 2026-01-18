from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_action', [None, 'ignore'])
def test_map_defaultdict_missing_key(na_action):
    s = Series([1, 2, np.nan])
    default_map = defaultdict(lambda: 'missing', {1: 'a', 2: 'b', 3: 'c'})
    result = s.map(default_map, na_action=na_action)
    expected = Series({0: 'a', 1: 'b', 2: 'missing' if na_action is None else np.nan})
    tm.assert_series_equal(result, expected)