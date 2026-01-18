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
@pytest.mark.parametrize('other_and_type', [('foo', 'str'), (2, 'int'), (2.0, 'float')])
@pytest.mark.parametrize('symbol_and_op', [('<=', operator.le), ('<', operator.lt), ('>=', operator.ge), ('>', operator.gt)])
def test_nat_comparisons_invalid(other_and_type, symbol_and_op):
    other, other_type = other_and_type
    symbol, op = symbol_and_op
    assert not NaT == other
    assert not other == NaT
    assert NaT != other
    assert other != NaT
    msg = f"'{symbol}' not supported between instances of 'NaTType' and '{other_type}'"
    with pytest.raises(TypeError, match=msg):
        op(NaT, other)
    msg = f"'{symbol}' not supported between instances of '{other_type}' and 'NaTType'"
    with pytest.raises(TypeError, match=msg):
        op(other, NaT)