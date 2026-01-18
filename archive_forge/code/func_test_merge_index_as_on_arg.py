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
def test_merge_index_as_on_arg(self, df, df2):
    left = df.set_index('key1')
    right = df2.set_index('key1')
    result = merge(left, right, on='key1')
    expected = merge(df, df2, on='key1').set_index('key1')
    tm.assert_frame_equal(result, expected)