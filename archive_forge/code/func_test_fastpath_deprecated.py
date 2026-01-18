from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_fastpath_deprecated(self):
    codes = np.array([1, 2, 3])
    dtype = CategoricalDtype(categories=['a', 'b', 'c', 'd'], ordered=False)
    msg = "The 'fastpath' keyword in Categorical is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        Categorical(codes, dtype=dtype, fastpath=True)