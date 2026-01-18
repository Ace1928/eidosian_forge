from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('func', ['merge', 'merge_asof'])
@pytest.mark.parametrize('kwargs', [{'right_index': True}, {'left_index': True}])
def test_merge_join_cols_error_reporting_on_and_index(func, kwargs):
    left = DataFrame({'a': [1, 2], 'b': [3, 4]})
    right = DataFrame({'a': [1, 1], 'c': [5, 6]})
    msg = 'Can only pass argument "on" OR "left_index" and "right_index", not a combination of both\\.'
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, on='a', **kwargs)