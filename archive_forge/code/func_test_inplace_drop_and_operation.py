import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('operation', ['__iadd__', '__isub__', '__imul__', '__ipow__'])
@pytest.mark.parametrize('inplace', [False, True])
def test_inplace_drop_and_operation(self, operation, inplace):
    df = DataFrame({'x': range(5)})
    expected = df.copy()
    df['y'] = range(5)
    y = df['y']
    with tm.assert_produces_warning(None):
        if inplace:
            df.drop('y', axis=1, inplace=inplace)
        else:
            df = df.drop('y', axis=1, inplace=inplace)
        getattr(y, operation)(1)
        tm.assert_frame_equal(df, expected)