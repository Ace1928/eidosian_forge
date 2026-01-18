from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_min_max_series(self):
    rng = date_range('1/1/2000', periods=10, freq='4h')
    lvls = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
    df = DataFrame({'TS': rng, 'V': np.random.default_rng(2).standard_normal(len(rng)), 'L': lvls})
    result = df.TS.max()
    exp = Timestamp(df.TS.iat[-1])
    assert isinstance(result, Timestamp)
    assert result == exp
    result = df.TS.min()
    exp = Timestamp(df.TS.iat[0])
    assert isinstance(result, Timestamp)
    assert result == exp