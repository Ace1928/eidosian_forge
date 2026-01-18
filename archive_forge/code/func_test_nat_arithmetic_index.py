from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
@pytest.mark.parametrize('op_name', ['left_plus_right', 'right_plus_left', 'left_minus_right', 'right_minus_left'])
@pytest.mark.parametrize('value', [DatetimeIndex(['2011-01-01', '2011-01-02'], name='x'), DatetimeIndex(['2011-01-01', '2011-01-02'], tz='US/Eastern', name='x'), DatetimeArray._from_sequence(['2011-01-01', '2011-01-02'], dtype='M8[ns]'), DatetimeArray._from_sequence(['2011-01-01', '2011-01-02'], dtype=DatetimeTZDtype(tz='US/Pacific')), TimedeltaIndex(['1 day', '2 day'], name='x')])
def test_nat_arithmetic_index(op_name, value):
    exp_name = 'x'
    exp_data = [NaT] * 2
    if value.dtype.kind == 'M' and 'plus' in op_name:
        expected = DatetimeIndex(exp_data, tz=value.tz, name=exp_name)
    else:
        expected = TimedeltaIndex(exp_data, name=exp_name)
    expected = expected.as_unit(value.unit)
    if not isinstance(value, Index):
        expected = expected.array
    op = _ops[op_name]
    result = op(NaT, value)
    tm.assert_equal(result, expected)