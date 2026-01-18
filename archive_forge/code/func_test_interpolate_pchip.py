import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interpolate_pchip(self):
    pytest.importorskip('scipy')
    ser = Series(np.sort(np.random.default_rng(2).uniform(size=100)))
    new_index = ser.index.union(Index([49.25, 49.5, 49.75, 50.25, 50.5, 50.75])).astype(float)
    interp_s = ser.reindex(new_index).interpolate(method='pchip')
    interp_s.loc[49:51]