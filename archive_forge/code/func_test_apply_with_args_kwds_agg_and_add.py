from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_with_args_kwds_agg_and_add(float_frame):

    def agg_and_add(x, howmuch=0):
        return x.mean() + howmuch
    result = float_frame.apply(agg_and_add, howmuch=2)
    expected = float_frame.apply(lambda x: x.mean() + 2)
    tm.assert_series_equal(result, expected)