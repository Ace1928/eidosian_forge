from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('axis', [0, 1])
def test_apply_raw_float_frame(float_frame, axis, engine):
    if engine == 'numba':
        pytest.skip("numba can't handle when UDF returns None.")

    def _assert_raw(x):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1
    float_frame.apply(_assert_raw, axis=axis, engine=engine, raw=True)