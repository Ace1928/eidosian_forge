import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_agg_nested_dicts():
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'two', 'two', 'two', 'one', 'two'], 'C': np.random.default_rng(2).standard_normal(8) + 1.0, 'D': np.arange(8)})
    g = df.groupby(['A', 'B'])
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        g.aggregate({'r1': {'C': ['mean', 'sum']}, 'r2': {'D': ['mean', 'sum']}})
    with pytest.raises(SpecificationError, match=msg):
        g.agg({'C': {'ra': ['mean', 'std']}, 'D': {'rb': ['mean', 'std']}})
    with pytest.raises(SpecificationError, match=msg):
        g['D'].agg({'result1': np.sum, 'result2': np.mean})
    with pytest.raises(SpecificationError, match=msg):
        g['D'].agg({'D': np.sum, 'result2': np.mean})