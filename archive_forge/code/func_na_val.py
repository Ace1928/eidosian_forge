import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def na_val(dtype):
    if dtype.storage == 'pyarrow_numpy':
        return np.nan
    else:
        return pd.NA