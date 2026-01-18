from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
def shifted_sum(df, before, after, c=0):
    a = df.shift(before)
    b = df.shift(-after)
    return df + a + b + c