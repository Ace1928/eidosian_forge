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
def test_merge_misspecified(self, df, df2, left, right):
    msg = 'Must pass right_on or right_index=True'
    with pytest.raises(pd.errors.MergeError, match=msg):
        merge(left, right, left_index=True)
    msg = 'Must pass left_on or left_index=True'
    with pytest.raises(pd.errors.MergeError, match=msg):
        merge(left, right, right_index=True)
    msg = 'Can only pass argument "on" OR "left_on" and "right_on", not a combination of both'
    with pytest.raises(pd.errors.MergeError, match=msg):
        merge(left, left, left_on='key', on='key')
    msg = 'len\\(right_on\\) must equal len\\(left_on\\)'
    with pytest.raises(ValueError, match=msg):
        merge(df, df2, left_on=['key1'], right_on=['key1', 'key2'])