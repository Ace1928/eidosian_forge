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
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('numeric_only', [True, False, None])
def test_datetime_std_with_larger_dataset(axis, skipna, numeric_only):
    num_rows = 250
    dt1 = pd.concat([pd.Series([pd.NaT] * 15, index=range(15)), pd.to_datetime(pd.Series([datetime.fromtimestamp(1636426704 + i * 250000) for i in range(num_rows - 15)], index=range(15, 250)))], ignore_index=False)
    base_numbers = [1638290040706793300 + i * 69527182702409 for i in range(num_rows)]
    pdf = pd.DataFrame({'dt1': dt1, 'dt2': pd.to_datetime(pd.Series(base_numbers))}, index=range(250))
    for i in range(3, 8):
        pdf[f'dt{i}'] = pd.to_datetime(pd.Series([int(x + 0.12 * i) for x in base_numbers]))
    ddf = dd.from_pandas(pdf, 8)
    kwargs = {} if numeric_only is None else {'numeric_only': numeric_only}
    kwargs['skipna'] = skipna
    expected = pdf[['dt1']].std(axis=axis, **kwargs)
    result = ddf[['dt1']].std(axis=axis, **kwargs)
    assert_near_timedeltas(ddf['dt1'].std(**kwargs).compute(), pdf['dt1'].std(**kwargs))
    expected = pdf.std(axis=axis, **kwargs)
    result = ddf.std(axis=axis, **kwargs)
    assert_near_timedeltas(result.compute(), expected)