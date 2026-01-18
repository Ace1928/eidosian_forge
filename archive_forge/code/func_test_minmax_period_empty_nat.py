from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('op', ['min', 'max'])
@pytest.mark.parametrize('data', [[], [NaT], [NaT, NaT, NaT]])
def test_minmax_period_empty_nat(self, op, data):
    obj = PeriodIndex(data, freq='M')
    result = getattr(obj, op)()
    assert result is NaT