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
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason='delayed not currently supported in here')
def test_groupby_shift_lazy_input():
    pdf = pd.DataFrame({'a': [0, 0, 1, 1, 2, 2, 3, 3, 3], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0], 'c': [0, 0, 0, 0, 0, 1, 1, 1, 1]})
    delayed_periods = dask.delayed(lambda: 1)()
    ddf = dd.from_pandas(pdf, npartitions=3)
    assert_eq(pdf.groupby(pdf.c).shift(periods=1), ddf.groupby(ddf.c).shift(periods=delayed_periods, meta={'a': int, 'b': int}))
    with pytest.warns(UserWarning):
        assert_eq(pdf.groupby(pdf.c).shift(periods=1, fill_value=pdf.b.max()), ddf.groupby(ddf.c).shift(periods=1, fill_value=ddf.b.max()))