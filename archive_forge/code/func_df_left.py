from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.fixture
def df_left():
    partition_sizes = np.array([3, 4, 2, 5, 3, 2, 5, 9, 4, 7, 4])
    idx = [i for i, s in enumerate(partition_sizes) for _ in range(s)]
    k = [i for s in partition_sizes for i in range(s)]
    vi = range(len(k))
    return pd.DataFrame(dict(idx=idx, k=k, v1=vi)).set_index(['idx'])