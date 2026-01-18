import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_level_values_when_periods():
    from pandas import Period, PeriodIndex
    idx = MultiIndex.from_arrays([PeriodIndex([Period('2019Q1'), Period('2019Q2')], name='b')])
    idx2 = MultiIndex.from_arrays([idx._get_level_values(level) for level in range(idx.nlevels)])
    assert all((x.is_monotonic_increasing for x in idx2.levels))