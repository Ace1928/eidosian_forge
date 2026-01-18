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
def test_categorize_nan():
    df = dd.from_pandas(pd.DataFrame({'A': ['a', 'b', 'a', float('nan')]}), npartitions=2)
    with warnings.catch_warnings(record=True) as record:
        df.categorize().compute()
    assert not record