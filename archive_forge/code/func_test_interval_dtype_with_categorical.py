import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
from pandas import (
@pytest.mark.parametrize('dtype', interval_dtypes)
def test_interval_dtype_with_categorical(dtype):
    obj = Index([], dtype=dtype)
    cat = Categorical([], categories=obj)
    result = find_common_type([dtype, cat.dtype])
    assert result == dtype