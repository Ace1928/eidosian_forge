from __future__ import annotations
import contextlib
import operator
import warnings
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import _concat
from dask.dataframe.utils import (
def test_categorical_accessor_presence():
    df = pd.DataFrame({'x': list('a' * 5 + 'b' * 5 + 'c' * 5), 'y': range(15)})
    df.x = df.x.astype('category')
    ddf = dd.from_pandas(df, npartitions=2)
    assert 'cat' in dir(ddf.x)
    assert 'cat' not in dir(ddf.y)
    assert hasattr(ddf.x, 'cat')
    assert not hasattr(ddf.y, 'cat')
    df2 = df.set_index(df.x)
    ddf2 = dd.from_pandas(df2, npartitions=2, sort=False)
    assert hasattr(ddf2.index, 'categories')
    assert not hasattr(ddf.index, 'categories')