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
@pytest.mark.parametrize('dtype,op,out_dtype', [('datetime64[ns]', operator.add, 'datetime64[ns]'), ('datetime64[ns]', roperator.radd, 'datetime64[ns]'), ('datetime64[ns]', operator.sub, 'timedelta64[ns]'), ('datetime64[ns]', roperator.rsub, 'timedelta64[ns]'), ('timedelta64[ns]', operator.add, 'datetime64[ns]'), ('timedelta64[ns]', roperator.radd, 'datetime64[ns]'), ('timedelta64[ns]', operator.sub, 'datetime64[ns]'), ('timedelta64[ns]', roperator.rsub, 'timedelta64[ns]')])
def test_nat_arithmetic_ndarray(dtype, op, out_dtype):
    other = np.arange(10).astype(dtype)
    result = op(NaT, other)
    expected = np.empty(other.shape, dtype=out_dtype)
    expected.fill('NaT')
    tm.assert_numpy_array_equal(result, expected)