from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_concat_preserve_names(three_group):
    grouped = three_group.groupby(['A', 'B'])

    def desc(group):
        result = group.describe()
        result.index.name = 'stat'
        return result

    def desc2(group):
        result = group.describe()
        result.index.name = 'stat'
        result = result[:len(group)]
        return result

    def desc3(group):
        result = group.describe()
        result.index.name = f'stat_{len(group):d}'
        result = result[:len(group)]
        return result
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = grouped.apply(desc)
    assert result.index.names == ('A', 'B', 'stat')
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result2 = grouped.apply(desc2)
    assert result2.index.names == ('A', 'B', 'stat')
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result3 = grouped.apply(desc3)
    assert result3.index.names == ('A', 'B', None)