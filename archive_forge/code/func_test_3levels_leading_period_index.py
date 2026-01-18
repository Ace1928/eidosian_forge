import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_3levels_leading_period_index():
    pi = pd.PeriodIndex(['20181101 1100', '20181101 1200', '20181102 1300', '20181102 1400'], name='datetime', freq='D')
    lev2 = ['A', 'A', 'Z', 'W']
    lev3 = ['B', 'C', 'Q', 'F']
    mi = MultiIndex.from_arrays([pi, lev2, lev3])
    ser = Series(range(4), index=mi, dtype=np.float64)
    result = ser.loc[pi[0], 'A', 'B']
    assert result == 0.0