from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_broadcast_list_lambda_func(int_frame_const_col):
    df = int_frame_const_col
    result = df.apply(lambda x: [1, 2, 3], axis=1, result_type='broadcast')
    tm.assert_frame_equal(result, df)