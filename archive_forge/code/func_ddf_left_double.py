from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.fixture
def ddf_left_double(df_left):
    return dd.from_pandas(df_left, npartitions=2, sort=False)