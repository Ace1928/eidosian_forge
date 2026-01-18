from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_attach_name_non_reduction(float_frame):
    result = float_frame.apply(lambda x: np.repeat(x.name, len(x)))
    expected = DataFrame(np.tile(float_frame.columns, (len(float_frame.index), 1)), index=float_frame.index, columns=float_frame.columns)
    tm.assert_frame_equal(result, expected)