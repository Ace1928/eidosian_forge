from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_multi_index():
    index = MultiIndex.from_arrays([['a', 'a', 'b'], ['c', 'd', 'd']])
    s = DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=['col1', 'col2'])
    result = s.apply(lambda x: Series({'min': min(x), 'max': max(x)}), 1)
    expected = DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=['min', 'max'])
    tm.assert_frame_equal(result, expected, check_like=True)