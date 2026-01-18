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
def test_merge_key_dtype_cast(self):
    df1 = DataFrame({'key': [1.0, 2.0], 'v1': [10, 20]}, columns=['key', 'v1'])
    df2 = DataFrame({'key': [2], 'v2': [200]}, columns=['key', 'v2'])
    result = df1.merge(df2, on='key', how='left')
    expected = DataFrame({'key': [1.0, 2.0], 'v1': [10, 20], 'v2': [np.nan, 200.0]}, columns=['key', 'v1', 'v2'])
    tm.assert_frame_equal(result, expected)