from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_logical_operators_int_dtype_with_str(self):
    s_1111 = Series([1] * 4, dtype='int8')
    warn_msg = 'Logical ops \\(and, or, xor\\) between Pandas objects and dtype-less sequences'
    msg = "Cannot perform 'and_' with a dtyped.+array and scalar of type"
    with pytest.raises(TypeError, match=msg):
        s_1111 & 'a'
    with pytest.raises(TypeError, match='unsupported operand.+for &'):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            s_1111 & ['a', 'b', 'c', 'd']