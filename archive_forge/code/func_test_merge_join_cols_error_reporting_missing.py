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
@pytest.mark.parametrize(('kwargs', 'err_msg'), [({'left_on': 'a'}, ['right_on', 'right_index']), ({'right_on': 'a'}, ['left_on', 'left_index'])])
def test_merge_join_cols_error_reporting_missing(func, kwargs, err_msg):
    left = DataFrame({'a': [1, 2], 'b': [3, 4]})
    right = DataFrame({'a': [1, 1], 'c': [5, 6]})
    msg = f'Must pass "{err_msg[0]}" OR "{err_msg[1]}"\\.'
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, **kwargs)