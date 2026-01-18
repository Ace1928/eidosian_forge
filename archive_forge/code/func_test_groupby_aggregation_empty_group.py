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
def test_groupby_aggregation_empty_group():

    def func(x):
        if len(x) == 0:
            raise ValueError('length must not be 0')
        return len(x)
    df = DataFrame({'A': pd.Categorical(['a', 'a'], categories=['a', 'b', 'c']), 'B': [1, 1]})
    msg = 'length must not be 0'
    with pytest.raises(ValueError, match=msg):
        df.groupby('A', observed=False).agg(func)