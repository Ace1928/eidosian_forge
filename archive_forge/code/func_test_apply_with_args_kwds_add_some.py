from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_with_args_kwds_add_some(float_frame):

    def add_some(x, howmuch=0):
        return x + howmuch
    result = float_frame.apply(add_some, howmuch=2)
    expected = float_frame.apply(lambda x: x + 2)
    tm.assert_frame_equal(result, expected)