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
def test_groupby_numeric_column():
    df = pd.DataFrame({'A': ['foo', 'foo', 'bar'], 0: [1, 2, 3]})
    ddf = dd.from_pandas(df, npartitions=3)
    assert_eq(ddf.groupby(ddf.A)[0].sum(), df.groupby(df.A)[0].sum())