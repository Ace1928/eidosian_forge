import numpy as np
import pytest
import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
def test_multiindex_perf_warn(self):
    df = DataFrame({'jim': [0, 0, 1, 1], 'joe': ['x', 'x', 'z', 'y'], 'jolie': np.random.default_rng(2).random(4)}).set_index(['jim', 'joe'])
    with tm.assert_produces_warning(PerformanceWarning):
        df.loc[1, 'z']
    df = df.iloc[[2, 1, 3, 0]]
    with tm.assert_produces_warning(PerformanceWarning):
        df.loc[0,]