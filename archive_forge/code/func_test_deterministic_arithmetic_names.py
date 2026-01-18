from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
def test_deterministic_arithmetic_names():
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8]})
    a = dd.from_pandas(df, npartitions=2)
    assert sorted((a.x + a.y ** 2).dask) == sorted((a.x + a.y ** 2).dask)
    assert sorted((a.x + a.y ** 2).dask) != sorted((a.x + a.y ** 3).dask)
    assert sorted((a.x + a.y ** 2).dask) != sorted((a.x - a.y ** 2).dask)