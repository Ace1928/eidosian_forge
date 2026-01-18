from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_categorical_func():
    df = DataFrame({'c0': ['A', 'A', 'B', 'B'], 'c1': ['C', 'C', 'D', 'D']})
    result = df.apply(lambda ts: ts.astype('category'))
    assert result.shape == (4, 2)
    assert isinstance(result['c0'].dtype, CategoricalDtype)
    assert isinstance(result['c1'].dtype, CategoricalDtype)