from __future__ import annotations
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
@pytest.mark.parametrize('check_scipy', [False, pytest.param(True, marks=td.skip_if_no('scipy'))])
def test_is_sparse(check_scipy):
    msg = 'is_sparse is deprecated'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert com.is_sparse(SparseArray([1, 2, 3]))
        assert not com.is_sparse(np.array([1, 2, 3]))
        if check_scipy:
            import scipy.sparse
            assert not com.is_sparse(scipy.sparse.bsr_matrix([1, 2, 3]))