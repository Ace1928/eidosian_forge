from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.fixture
def ddf_right(df_right):
    return dd.repartition(df_right, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11])