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
def test_groupby_agg_grouper_single():
    d = pd.DataFrame({'a': [1, 2, 3, 4]})
    a = dd.from_pandas(d, npartitions=2)
    result = a.groupby('a')['a'].agg(['min', 'max'])
    expected = d.groupby('a')['a'].agg(['min', 'max'])
    assert_eq(result, expected)