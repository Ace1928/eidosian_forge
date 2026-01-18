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
def test_assert_eq_scheduler():
    using_custom_scheduler = False

    def custom_scheduler(*args, **kwargs):
        nonlocal using_custom_scheduler
        try:
            using_custom_scheduler = True
            return get_sync(*args, **kwargs)
        finally:
            using_custom_scheduler = False

    def check_custom_scheduler(part: pd.DataFrame) -> pd.DataFrame:
        assert using_custom_scheduler, 'not using custom scheduler'
        return part + 1
    df = pd.DataFrame({'x': [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=2)
    ddf2 = ddf.map_partitions(check_custom_scheduler, meta=ddf)
    with pytest.raises(AssertionError, match='not using custom scheduler'):
        assert_eq(ddf2, ddf2)
    assert_eq(ddf2, ddf2, scheduler=custom_scheduler)
    with dask.config.set(scheduler=custom_scheduler):
        assert_eq(ddf2, ddf2, scheduler=None)