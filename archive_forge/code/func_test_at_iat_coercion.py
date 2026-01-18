from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_at_iat_coercion(self):
    dates = date_range('1/1/2000', periods=8)
    df = DataFrame(np.random.default_rng(2).standard_normal((8, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
    s = df['A']
    result = s.at[dates[5]]
    xp = s.values[5]
    assert result == xp