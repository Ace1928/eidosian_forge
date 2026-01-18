import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('func', [['min'], ['mean', 'max'], {'b': 'sum'}, {'b': 'prod', 'c': 'median'}])
def test_multi_axis_1_raises(func):
    df = DataFrame({'a': [1, 1, 2], 'b': [3, 4, 5], 'c': [6, 7, 8]})
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby('a', axis=1)
    with pytest.raises(NotImplementedError, match='axis other than 0 is not supported'):
        gb.agg(func)