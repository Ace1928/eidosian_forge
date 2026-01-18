import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_datatype_categorical_error_raises(self):
    msg = 'incompatible merge keys \\[0\\] .* both sides category, but not equal ones'
    left = pd.DataFrame({'left_val': [1, 5, 10], 'a': pd.Categorical(['a', 'b', 'c'])})
    right = pd.DataFrame({'right_val': [1, 2, 3, 6, 7], 'a': pd.Categorical(['a', 'X', 'c', 'X', 'b'])})
    with pytest.raises(MergeError, match=msg):
        merge_asof(left, right, on='a')