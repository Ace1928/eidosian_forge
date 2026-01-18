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
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('numeric_only', [True, False, None])
def test_datetime_std_across_axis1_null_results(skipna, numeric_only):
    pdf = pd.DataFrame({'dt1': [datetime.fromtimestamp(1636426704 + i * 250000) for i in range(10)], 'dt2': [datetime.fromtimestamp(1636426704 + i * 217790) for i in range(10)], 'nums': [i for i in range(10)]})
    ddf = dd.from_pandas(pdf, 3)
    kwargs = {} if numeric_only is None else {'numeric_only': numeric_only}
    kwargs['skipna'] = skipna
    ctx = contextlib.nullcontext()
    success = True
    if numeric_only is False or (PANDAS_GE_200 and numeric_only is None):
        ctx = pytest.raises(TypeError)
        success = False
    elif numeric_only is None:
        ctx = pytest.warns(FutureWarning, match='numeric_only')
    expected = pdf[['dt1']].std(axis=1, **kwargs)
    result = ddf[['dt1']].std(axis=1, **kwargs)
    if success:
        assert_eq(result, expected)
    with ctx:
        expected = pdf.std(axis=1, **kwargs)
    with ctx:
        result = ddf.std(axis=1, **kwargs)
    if success:
        assert_eq(result, expected)
    pdf2 = pd.DataFrame({'dt1': [pd.NaT] + [datetime.fromtimestamp(1636426704 + i * 250000) for i in range(10)] + [pd.NaT], 'dt2': [datetime.fromtimestamp(1636426704 + i * 250000) for i in range(12)], 'dt3': [datetime.fromtimestamp(1636426704 + i * 282616) for i in range(12)]})
    ddf2 = dd.from_pandas(pdf2, 3)
    expected = pdf2.std(axis=1, **kwargs)
    result = ddf2.std(axis=1, **kwargs)
    if success:
        assert_eq(result, expected)