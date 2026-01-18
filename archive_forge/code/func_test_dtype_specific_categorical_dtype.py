import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_dtype_specific_categorical_dtype(self):
    expected = 'datetime64[ns]'
    dti = DatetimeIndex([], dtype=expected)
    result = str(Categorical(dti).categories.dtype)
    assert result == expected