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
def test_frame_groupby(tsframe):
    grouped = tsframe.groupby(lambda x: x.weekday())
    aggregated = grouped.aggregate('mean')
    assert len(aggregated) == 5
    assert len(aggregated.columns) == 4
    tscopy = tsframe.copy()
    tscopy['weekday'] = [x.weekday() for x in tscopy.index]
    stragged = tscopy.groupby('weekday').aggregate('mean')
    tm.assert_frame_equal(stragged, aggregated, check_names=False)
    grouped = tsframe.head(30).groupby(lambda x: x.weekday())
    transformed = grouped.transform(lambda x: x - x.mean())
    assert len(transformed) == 30
    assert len(transformed.columns) == 4
    transformed = grouped.transform(lambda x: x.mean())
    for name, group in grouped:
        mean = group.mean()
        for idx in group.index:
            tm.assert_series_equal(transformed.xs(idx), mean, check_names=False)
    for weekday, group in grouped:
        assert group.index[0].weekday() == weekday
    groups = grouped.groups
    indices = grouped.indices
    for k, v in groups.items():
        samething = tsframe.index.take(indices[k])
        assert (samething == v).all()