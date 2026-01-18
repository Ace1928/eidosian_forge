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
def assert_near_timedeltas(t1, t2, atol=2000):
    if is_scalar(t1):
        t1 = pd.Series([t1])
    if is_scalar(t2):
        t2 = pd.Series([t2])
    assert t1.dtype == t2.dtype
    assert_eq(pd.to_numeric(t1), pd.to_numeric(t2), atol=atol)