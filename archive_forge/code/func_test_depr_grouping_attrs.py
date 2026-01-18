from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('attr', ['group_index', 'result_index', 'group_arraylike'])
def test_depr_grouping_attrs(attr):
    df = DataFrame({'a': [1, 1, 2], 'b': [3, 4, 5]})
    gb = df.groupby('a')
    msg = f'{attr} is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        getattr(gb._grouper.groupings[0], attr)