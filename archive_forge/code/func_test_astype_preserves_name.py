from copy import (
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int64', 'uint64', 'float64', 'category', 'datetime64[ns]', 'timedelta64[ns]'])
def test_astype_preserves_name(self, index, dtype):
    if isinstance(index, MultiIndex):
        index.names = ['idx' + str(i) for i in range(index.nlevels)]
    else:
        index.name = 'idx'
    warn = None
    if index.dtype.kind == 'c' and dtype in ['float64', 'int64', 'uint64']:
        if np_version_gte1p25:
            warn = np.exceptions.ComplexWarning
        else:
            warn = np.ComplexWarning
    is_pyarrow_str = str(index.dtype) == 'string[pyarrow]' and dtype == 'category'
    try:
        with tm.assert_produces_warning(warn, raise_on_extra_warnings=is_pyarrow_str, check_stacklevel=False):
            result = index.astype(dtype)
    except (ValueError, TypeError, NotImplementedError, SystemError):
        return
    if isinstance(index, MultiIndex):
        assert result.names == index.names
    else:
        assert result.name == index.name