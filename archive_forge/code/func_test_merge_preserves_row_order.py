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
@pytest.mark.parametrize('how', ['left', 'right'])
def test_merge_preserves_row_order(self, how):
    left_df = DataFrame({'animal': ['dog', 'pig'], 'max_speed': [40, 11]})
    right_df = DataFrame({'animal': ['quetzal', 'pig'], 'max_speed': [80, 11]})
    result = left_df.merge(right_df, on=['animal', 'max_speed'], how=how)
    if how == 'right':
        expected = DataFrame({'animal': ['quetzal', 'pig'], 'max_speed': [80, 11]})
    else:
        expected = DataFrame({'animal': ['dog', 'pig'], 'max_speed': [40, 11]})
    tm.assert_frame_equal(result, expected)