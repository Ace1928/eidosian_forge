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
@pytest.mark.parametrize('func', ['sum', 'count', 'mean', 'var', 'sem'])
def test_empty_df_reductions(func):
    pdf = pd.DataFrame()
    ddf = dd.from_pandas(pdf, npartitions=1)
    dsk_func = getattr(ddf.__class__, func)
    pd_func = getattr(pdf.__class__, func)
    assert_eq(dsk_func(ddf), pd_func(pdf))
    idx = pd.date_range('2000', periods=4)
    pdf = pd.DataFrame(index=idx)
    ddf = dd.from_pandas(pdf, npartitions=1)
    assert_eq(dsk_func(ddf), pd_func(pdf))