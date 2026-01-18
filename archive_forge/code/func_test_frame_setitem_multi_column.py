import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_frame_setitem_multi_column(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=[['a', 'a', 'b', 'b'], [0, 1, 0, 1]])
    cp = df.copy()
    cp['a'] = cp['b']
    tm.assert_frame_equal(cp['a'], cp['b'])
    cp = df.copy()
    cp['a'] = cp['b'].values
    tm.assert_frame_equal(cp['a'], cp['b'])