from __future__ import annotations
import re
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_300, tm
from dask.dataframe.core import apply_and_enforce
from dask.dataframe.utils import (
from dask.local import get_sync
def test_check_meta_typename():
    df = pd.DataFrame({'x': []})
    ddf = dd.from_pandas(df, npartitions=1)
    check_meta(df, df)
    with pytest.raises(Exception) as info:
        check_meta(ddf, df)
    assert 'dask' in str(info.value)
    assert 'pandas' in str(info.value)