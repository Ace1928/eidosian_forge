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
@pytest.mark.parametrize('split_every', [False, 2])
@pytest.mark.xfail_with_pyarrow_strings
def test_allany(split_every):
    df = pd.DataFrame(np.random.choice([True, False], size=(100, 4)), columns=['A', 'B', 'C', 'D'])
    df['E'] = list('abcde') * 20
    ddf = dd.from_pandas(df, 10)
    assert_eq(ddf.all(split_every=split_every), df.all())
    assert_eq(ddf.all(axis=1, split_every=split_every), df.all(axis=1))
    assert_eq(ddf.all(axis=0, split_every=split_every), df.all(axis=0))
    assert_eq(ddf.any(split_every=split_every), df.any())
    assert_eq(ddf.any(axis=1, split_every=split_every), df.any(axis=1))
    assert_eq(ddf.any(axis=0, split_every=split_every), df.any(axis=0))
    assert_eq(ddf.A.all(split_every=split_every), df.A.all())
    assert_eq(ddf.A.any(split_every=split_every), df.A.any())
    ddf_out_axis_default = dd.from_pandas(pd.Series([False, False, False, False, False], index=['A', 'B', 'C', 'D', 'E']), 10)
    ddf_out_axis1 = dd.from_pandas(pd.Series(np.random.choice([True, False], size=(100,))), 10)
    with pytest.warns(FutureWarning, match="the 'out' keyword is deprecated"):
        ddf.all(split_every=split_every, out=ddf_out_axis_default)
    assert_eq(ddf_out_axis_default, df.all())
    with pytest.warns(FutureWarning, match="the 'out' keyword is deprecated"):
        ddf.all(axis=1, split_every=split_every, out=ddf_out_axis1)
    assert_eq(ddf_out_axis1, df.all(axis=1))
    with pytest.warns(FutureWarning, match="the 'out' keyword is deprecated"):
        ddf.all(split_every=split_every, axis=0, out=ddf_out_axis_default)
    assert_eq(ddf_out_axis_default, df.all(axis=0))
    with pytest.warns(FutureWarning, match="the 'out' keyword is deprecated"):
        ddf.any(split_every=split_every, out=ddf_out_axis_default)
    assert_eq(ddf_out_axis_default, df.any())
    with pytest.warns(FutureWarning, match="the 'out' keyword is deprecated"):
        ddf.any(axis=1, split_every=split_every, out=ddf_out_axis1)
    assert_eq(ddf_out_axis1, df.any(axis=1))
    with pytest.warns(FutureWarning, match="the 'out' keyword is deprecated"):
        ddf.any(split_every=split_every, axis=0, out=ddf_out_axis_default)
    assert_eq(ddf_out_axis_default, df.any(axis=0))