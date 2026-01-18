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
@pytest.mark.parametrize('agg', [lambda grp: grp.agg(partial(np.std, ddof=1)), lambda grp: grp.agg(partial(np.std, ddof=-2)), lambda grp: grp.agg(partial(np.var, ddof=1)), lambda grp: grp.agg(partial(np.var, ddof=-2))])
def test_groupby_aggregate_partial_function(agg):
    pdf = pd.DataFrame({'a': [5, 4, 3, 5, 4, 2, 3, 2], 'b': [1, 2, 5, 6, 9, 2, 6, 8]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    assert_eq(agg(pdf.groupby('a')), agg(ddf.groupby('a')))
    assert_eq(agg(pdf.groupby('a')['b']), agg(ddf.groupby('a')['b']))