import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('other', [1.0, np.array(1.0)])
def test_arithmetic_conversion(all_arithmetic_operators, other):
    op = tm.get_op_from_name(all_arithmetic_operators)
    s = pd.Series([1, 2, 3], dtype='Int64')
    result = op(s, other)
    assert result.dtype == 'Float64'