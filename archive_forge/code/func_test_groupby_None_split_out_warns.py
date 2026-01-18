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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='will raise')
def test_groupby_None_split_out_warns():
    df = pd.DataFrame({'a': [1, 1, 2], 'b': [2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=1)
    with pytest.warns(FutureWarning, match='split_out=None'):
        ddf.groupby('a').agg({'b': 'max'}, split_out=None)