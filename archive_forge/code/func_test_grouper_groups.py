from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouper_groups():
    df = DataFrame({'a': [1, 2, 3], 'b': 1})
    grper = Grouper(key='a')
    gb = df.groupby(grper)
    msg = 'Use GroupBy.groups instead'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = grper.groups
    assert res is gb.groups
    msg = 'Use GroupBy.grouper instead'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = grper.grouper
    assert res is gb._grouper
    msg = 'Grouper.obj is deprecated and will be removed'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = grper.obj
    assert res is gb.obj
    msg = 'Use Resampler.ax instead'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grper.ax
    msg = 'Grouper.indexer is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grper.indexer