import sys
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import PYPY
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('op_name, op', [('add', '+'), ('sub', '-'), ('mul', '*'), ('mod', '%'), ('pow', '**'), ('truediv', '/'), ('floordiv', '//')])
def test_binary_ops_docstring(frame_or_series, op_name, op):
    klass = frame_or_series
    operand1 = klass.__name__.lower()
    operand2 = 'other'
    expected_str = ' '.join([operand1, op, operand2])
    assert expected_str in getattr(klass, op_name).__doc__
    expected_str = ' '.join([operand2, op, operand1])
    assert expected_str in getattr(klass, 'r' + op_name).__doc__