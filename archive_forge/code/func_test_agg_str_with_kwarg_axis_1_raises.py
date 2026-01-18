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
def test_agg_str_with_kwarg_axis_1_raises(df, reduction_func):
    gb = df.groupby(level=0)
    warn_msg = f'DataFrameGroupBy.{reduction_func} with axis=1 is deprecated'
    if reduction_func in ('idxmax', 'idxmin'):
        error = TypeError
        msg = "'[<>]' not supported between instances of 'float' and 'str'"
        warn = FutureWarning
    else:
        error = ValueError
        msg = f'Operation {reduction_func} does not support axis=1'
        warn = None
    with pytest.raises(error, match=msg):
        with tm.assert_produces_warning(warn, match=warn_msg):
            gb.agg(reduction_func, axis=1)