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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='FIXME hangs')
def test_groupby_iter_fails():
    df = pd.DataFrame(data=[['a0', 'b1'], ['a1', 'b1'], ['a3', 'b3'], ['a5', 'b5']], columns=['A', 'B'])
    ddf = dd.from_pandas(df, npartitions=1)
    with pytest.raises(NotImplementedError, match='computing the groups'):
        list(ddf.groupby('A'))