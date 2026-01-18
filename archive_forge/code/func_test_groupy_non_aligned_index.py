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
def test_groupy_non_aligned_index():
    pdf = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7] * 10, 'b': [4, 2, 7, 3, 3, 1, 1, 1, 2] * 10, 'c': [0, 1, 2, 3, 4, 5, 6, 7, 8] * 10}, columns=['c', 'b', 'a'])
    ddf3 = dd.from_pandas(pdf, npartitions=3)
    ddf7 = dd.from_pandas(pdf, npartitions=7)
    ddf3.groupby(['a', 'b'])
    ddf3.groupby([ddf3['a'], ddf3['b']])
    with pytest.raises(NotImplementedError):
        ddf3.groupby(ddf7['a'])
    with pytest.raises(NotImplementedError):
        ddf3.groupby([ddf7['a'], ddf7['b']])
    with pytest.raises(NotImplementedError):
        ddf3.groupby([ddf7['a'], ddf3['b']])
    with pytest.raises(NotImplementedError):
        ddf3.groupby([ddf3['a'], ddf7['b']])
    with pytest.raises(NotImplementedError):
        ddf3.groupby([ddf7['a'], 'b'])