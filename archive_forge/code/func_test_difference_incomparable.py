from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('opname', ['difference', 'symmetric_difference'])
def test_difference_incomparable(self, opname):
    a = Index([3, Timestamp('2000'), 1])
    b = Index([2, Timestamp('1999'), 1])
    op = operator.methodcaller(opname, b)
    with tm.assert_produces_warning(RuntimeWarning):
        result = op(a)
    expected = Index([3, Timestamp('2000'), 2, Timestamp('1999')])
    if opname == 'difference':
        expected = expected[:2]
    tm.assert_index_equal(result, expected)
    op = operator.methodcaller(opname, b, sort=False)
    result = op(a)
    tm.assert_index_equal(result, expected)