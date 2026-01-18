from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
def test_modulo2(self):
    with np.errstate(all='ignore'):
        p = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
        result = p['first'] % p['second']
        expected = Series(p['first'].values % p['second'].values, dtype='float64')
        expected.iloc[0:3] = np.nan
        tm.assert_series_equal(result, expected)
        result = p['first'] % 0
        expected = Series(np.nan, index=p.index, name='first')
        tm.assert_series_equal(result, expected)
        p = p.astype('float64')
        result = p['first'] % p['second']
        expected = Series(p['first'].values % p['second'].values)
        tm.assert_series_equal(result, expected)
        p = p.astype('float64')
        result = p['first'] % p['second']
        result2 = p['second'] % p['first']
        assert not result.equals(result2)