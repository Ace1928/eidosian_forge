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
@pytest.mark.parametrize('func', ['var', 'std'])
@pytest.mark.parametrize('observed', [True, False])
@pytest.mark.parametrize('dropna', [True, False])
def test_groupby_var_dropna_observed(dropna, observed, func):
    df = pd.DataFrame({'a': [11, 12, 31, 1, 2, 3, 4, 5, 6, 10], 'b': pd.Categorical(values=[1] * 9 + [np.nan], categories=[1, 2])})
    ddf = dd.from_pandas(df, npartitions=3)
    dd_result = getattr(ddf.groupby('b', observed=observed, dropna=dropna), func)()
    pdf_result = getattr(df.groupby('b', observed=observed, dropna=dropna), func)()
    assert_eq(dd_result, pdf_result)