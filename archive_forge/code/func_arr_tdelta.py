from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.fixture
def arr_tdelta(arr_shape):
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype('m8[ns]')