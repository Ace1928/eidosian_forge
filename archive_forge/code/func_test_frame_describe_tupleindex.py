import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_frame_describe_tupleindex():
    df1 = DataFrame({'x': [1, 2, 3, 4, 5] * 3, 'y': [10, 20, 30, 40, 50] * 3, 'z': [100, 200, 300, 400, 500] * 3})
    df1['k'] = [(0, 0, 1), (0, 1, 0), (1, 0, 0)] * 5
    df2 = df1.rename(columns={'k': 'key'})
    msg = 'Names should be list-like for a MultiIndex'
    with pytest.raises(ValueError, match=msg):
        df1.groupby('k').describe()
    with pytest.raises(ValueError, match=msg):
        df2.groupby('key').describe()