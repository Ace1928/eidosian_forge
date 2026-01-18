from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
def test_accessor_works():
    with ensure_removed(dd.Series, 'mine'):
        dd.extensions.register_series_accessor('mine')(MyAccessor)
        a = pd.Series([1, 2])
        b = dd.from_pandas(a, 2)
        assert b.mine.obj is b
        assert b.mine.prop == 'item'
        assert b.mine.method() == 'item'