from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_logical_operators_int_dtype_with_float(self):
    s_0123 = Series(range(4), dtype='int64')
    warn_msg = 'Logical ops \\(and, or, xor\\) between Pandas objects and dtype-less sequences'
    msg = 'Cannot perform.+with a dtyped.+array and scalar of type'
    with pytest.raises(TypeError, match=msg):
        s_0123 & np.nan
    with pytest.raises(TypeError, match=msg):
        s_0123 & 3.14
    msg = 'unsupported operand type.+for &:'
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            s_0123 & [0.1, 4, 3.14, 2]
    with pytest.raises(TypeError, match=msg):
        s_0123 & np.array([0.1, 4, 3.14, 2])
    with pytest.raises(TypeError, match=msg):
        s_0123 & Series([0.1, 4, -3.14, 2])