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
@pytest.mark.parametrize('skipna', [True, False])
def test_df_groupby_idxmax_skipna(skipna):
    pdf = pd.DataFrame({'idx': list(range(4)), 'group': [1, 1, 2, 2], 'value': [np.nan, 20.1, np.nan, 10.1]}).set_index('idx')
    ddf = dd.from_pandas(pdf, npartitions=2)
    result_dd = ddf.groupby('group').idxmax(skipna=skipna)
    ctx = contextlib.nullcontext()
    if not skipna and PANDAS_GE_210 and (not PANDAS_GE_300):
        ctx = pytest.warns(FutureWarning, match='all-NA values')
    elif not skipna and PANDAS_GE_300:
        ctx = pytest.raises(ValueError, match='encountered an NA')
    with ctx:
        result_pd = pdf.groupby('group').idxmax(skipna=skipna)
    if not skipna and PANDAS_GE_300:
        return
    with ctx:
        assert_eq(result_pd, result_dd)