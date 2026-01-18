import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_agg_callables():
    df = DataFrame({'foo': [1, 2], 'bar': [3, 4]}).astype(np.int64)

    class fn_class:

        def __call__(self, x):
            return sum(x)
    equiv_callables = [sum, np.sum, lambda x: sum(x), lambda x: x.sum(), partial(sum), fn_class()]
    expected = df.groupby('foo').agg('sum')
    for ecall in equiv_callables:
        warn = FutureWarning if ecall is sum or ecall is np.sum else None
        msg = 'using DataFrameGroupBy.sum'
        with tm.assert_produces_warning(warn, match=msg):
            result = df.groupby('foo').agg(ecall)
        tm.assert_frame_equal(result, expected)