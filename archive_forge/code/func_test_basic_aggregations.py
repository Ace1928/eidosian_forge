from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
@pytest.mark.parametrize('dtype', ['int64', 'int32', 'float64', 'float32'])
def test_basic_aggregations(dtype):
    data = Series(np.arange(9) // 3, index=np.arange(9), dtype=dtype)
    index = np.arange(9)
    np.random.default_rng(2).shuffle(index)
    data = data.reindex(index)
    grouped = data.groupby(lambda x: x // 3, group_keys=False)
    for k, v in grouped:
        assert len(v) == 3
    msg = 'using SeriesGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        agged = grouped.aggregate(np.mean)
    assert agged[1] == 1
    msg = 'using SeriesGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = grouped.agg(np.mean)
    tm.assert_series_equal(agged, expected)
    tm.assert_series_equal(agged, grouped.mean())
    result = grouped.sum()
    msg = 'using SeriesGroupBy.sum'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = grouped.agg(np.sum)
    tm.assert_series_equal(result, expected)
    expected = grouped.apply(lambda x: x * x.sum())
    transformed = grouped.transform(lambda x: x * x.sum())
    assert transformed[7] == 12
    tm.assert_series_equal(transformed, expected)
    value_grouped = data.groupby(data)
    msg = 'using SeriesGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = value_grouped.aggregate(np.mean)
    tm.assert_series_equal(result, agged, check_index_type=False)
    msg = 'using SeriesGroupBy.[mean|std]'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        agged = grouped.aggregate([np.mean, np.std])
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate({'one': np.mean, 'two': np.std})
    group_constants = {0: 10, 1: 20, 2: 30}
    msg = 'Pinning the groupby key to each group in SeriesGroupBy.agg is deprecated, and cases that relied on it will raise in a future version'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        agged = grouped.agg(lambda x: group_constants[x.name] + x.mean())
    assert agged[1] == 21
    msg = 'Must produce aggregated value'
    with pytest.raises(Exception, match=msg):
        grouped.aggregate(lambda x: x * 2)