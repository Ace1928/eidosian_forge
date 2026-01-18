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
def test_groupby_string_label():
    df = pd.DataFrame({'foo': [1, 1, 4], 'B': [2, 3, 4], 'C': [5, 6, 7]})
    ddf = dd.from_pandas(pd.DataFrame(df), npartitions=1)
    ddf_group = ddf.groupby('foo')
    result = ddf_group.get_group(1).compute()
    expected = pd.DataFrame({'foo': [1, 1], 'B': [2, 3], 'C': [5, 6]}, index=pd.Index([0, 1]))
    tm.assert_frame_equal(result, expected)