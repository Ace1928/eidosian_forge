from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='axis not supported')
@pytest.mark.parametrize('func', ['idxmin', 'idxmax'])
@pytest.mark.parametrize('axis', [0, 1, 'index', 'columns'])
def test_df_groupby_idx_axis(func, axis):
    pdf = pd.DataFrame({'idx': list(range(4)), 'group': [1, 1, 2, 2], 'value': [10, 20, 20, 10]}).set_index('idx')
    ddf = dd.from_pandas(pdf, npartitions=2)
    warn = None if DASK_EXPR_ENABLED else FutureWarning
    if axis in (1, 'columns'):
        with pytest.raises(NotImplementedError), pytest.warns(warn, match='`axis` parameter is deprecated'):
            getattr(ddf.groupby('group'), func)(axis=axis)
    else:
        with groupby_axis_deprecated(dask_op=False):
            expected = getattr(pdf.groupby('group'), func)(axis=axis)
        deprecate_ctx = pytest.warns(warn, match='`axis` parameter is deprecated')
        with groupby_axis_deprecated(contextlib.nullcontext() if PANDAS_GE_210 else deprecate_ctx):
            result = getattr(ddf.groupby('group'), func)(axis=axis)
        assert_eq(expected, result)