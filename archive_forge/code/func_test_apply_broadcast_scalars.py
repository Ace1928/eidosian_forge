from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_broadcast_scalars(float_frame):
    result = float_frame.apply(np.mean, result_type='broadcast')
    expected = DataFrame([float_frame.mean()], index=float_frame.index)
    tm.assert_frame_equal(result, expected)