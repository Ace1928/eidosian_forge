from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_minmax_period(self):
    idx1 = PeriodIndex([NaT, '2011-01-01', '2011-01-02', '2011-01-03'], freq='D')
    assert not idx1.is_monotonic_increasing
    assert idx1[1:].is_monotonic_increasing
    idx2 = PeriodIndex(['2011-01-01', NaT, '2011-01-03', '2011-01-02', NaT], freq='D')
    assert not idx2.is_monotonic_increasing
    for idx in [idx1, idx2]:
        assert idx.min() == Period('2011-01-01', freq='D')
        assert idx.max() == Period('2011-01-03', freq='D')
    assert idx1.argmin() == 1
    assert idx2.argmin() == 0
    assert idx1.argmax() == 3
    assert idx2.argmax() == 2