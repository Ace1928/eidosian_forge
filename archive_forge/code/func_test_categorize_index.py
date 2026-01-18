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
def test_categorize_index():
    pdf = _compat.makeDataFrame()
    ddf = dd.from_pandas(pdf, npartitions=5)
    result = ddf.compute()
    ddf2 = ddf.categorize()
    assert ddf2.index.cat.known
    assert_eq(ddf2, result.set_index(pd.CategoricalIndex(result.index)), check_divisions=False, check_categorical=False)
    assert ddf.categorize(index=False) is ddf
    ddf = dd.from_pandas(result.set_index(result.A.rename('idx')), npartitions=5)
    result = ddf.compute()
    ddf2 = ddf.categorize(index=True)
    assert ddf2.index.cat.known
    assert_eq(ddf2, result.set_index(pd.CategoricalIndex(result.index)), check_divisions=False, check_categorical=False)
    assert ddf.categorize() is ddf