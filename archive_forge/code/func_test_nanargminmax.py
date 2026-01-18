from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('opname', ['max', 'min'])
def test_nanargminmax(self, opname, index_or_series):
    klass = index_or_series
    arg_op = 'arg' + opname if klass is Index else 'idx' + opname
    obj = klass([NaT, datetime(2011, 11, 1)])
    assert getattr(obj, arg_op)() == 1
    msg = 'The behavior of (DatetimeIndex|Series).argmax/argmin with skipna=False and NAs'
    if klass is Series:
        msg = 'The behavior of Series.(idxmax|idxmin) with all-NA'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(obj, arg_op)(skipna=False)
    if klass is Series:
        assert np.isnan(result)
    else:
        assert result == -1
    obj = klass([NaT, datetime(2011, 11, 1), NaT])
    assert getattr(obj, arg_op)() == 1
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(obj, arg_op)(skipna=False)
    if klass is Series:
        assert np.isnan(result)
    else:
        assert result == -1