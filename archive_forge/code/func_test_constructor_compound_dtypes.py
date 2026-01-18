from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_compound_dtypes(self):

    def f(dtype):
        data = list(itertools.repeat((datetime(2001, 1, 1), 'aa', 20), 9))
        return DataFrame(data=data, columns=['A', 'B', 'C'], dtype=dtype)
    msg = 'compound dtypes are not implemented in the DataFrame constructor'
    with pytest.raises(NotImplementedError, match=msg):
        f([('A', 'datetime64[h]'), ('B', 'str'), ('C', 'int32')])
    with pytest.raises(TypeError, match='argument must be'):
        f('int64')
    with pytest.raises(TypeError, match='argument must be'):
        f('float64')
    msg = '^Unknown datetime string format, unable to parse: aa, at position 0$'
    with pytest.raises(ValueError, match=msg):
        f('M8[ns]')