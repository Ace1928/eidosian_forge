from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
@pytest.mark.parametrize('spmatrix_t', ['coo_matrix', 'csc_matrix', 'csr_matrix'])
@pytest.mark.parametrize('dtype', [np.int64, np.float64, complex])
def test_loc_getitem_range_from_spmatrix(self, spmatrix_t, dtype):
    sp_sparse = pytest.importorskip('scipy.sparse')
    spmatrix_t = getattr(sp_sparse, spmatrix_t)
    rows, cols = (5, 7)
    spmatrix = spmatrix_t(np.eye(rows, cols, dtype=dtype), dtype=dtype)
    df = DataFrame.sparse.from_spmatrix(spmatrix)
    itr_idx = range(2, rows)
    result = df.loc[itr_idx].values
    expected = spmatrix.toarray()[itr_idx]
    tm.assert_numpy_array_equal(result, expected)
    result = df.loc[itr_idx].dtypes.values
    expected = np.full(cols, SparseDtype(dtype, fill_value=0))
    tm.assert_numpy_array_equal(result, expected)