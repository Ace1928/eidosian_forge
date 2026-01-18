import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_float_index_to_mixed(self):
    df = DataFrame({0.0: np.random.default_rng(2).random(10), 1.0: np.random.default_rng(2).random(10)})
    df['a'] = 10
    expected = DataFrame({0.0: df[0.0], 1.0: df[1.0], 'a': [10] * 10})
    tm.assert_frame_equal(expected, df)