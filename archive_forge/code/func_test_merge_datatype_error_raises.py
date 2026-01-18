import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_datatype_error_raises(self, using_infer_string):
    if using_infer_string:
        msg = 'incompatible merge keys'
    else:
        msg = 'Incompatible merge dtype, .*, both sides must have numeric dtype'
    left = pd.DataFrame({'left_val': [1, 5, 10], 'a': ['a', 'b', 'c']})
    right = pd.DataFrame({'right_val': [1, 2, 3, 6, 7], 'a': [1, 2, 3, 6, 7]})
    with pytest.raises(MergeError, match=msg):
        merge_asof(left, right, on='a')