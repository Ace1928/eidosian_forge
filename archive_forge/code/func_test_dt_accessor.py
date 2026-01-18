from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
@pytest.mark.skipif(not PANDAS_GE_210 or PANDAS_GE_300, reason='warning is None|divisions are incorrect')
def test_dt_accessor(df_ddf):
    df, ddf = df_ddf
    assert 'date' in dir(ddf.dt_col.dt)
    assert_eq(ddf.dt_col.dt.date, df.dt_col.dt.date, check_names=False)
    warning_ctx = pytest.warns(FutureWarning, match='will return a Series')
    with warning_ctx:
        ddf_result = ddf.dt_col.dt.to_pydatetime()
    with warning_ctx:
        pd_result = pd.Series(df.dt_col.dt.to_pydatetime(), index=df.index, dtype=object)
    assert_eq(ddf_result, pd_result)
    assert set(ddf.dt_col.dt.date.dask) == set(ddf.dt_col.dt.date.dask)
    if dd._dask_expr_enabled():
        ctx = contextlib.nullcontext()
    with ctx:
        assert set(ddf.dt_col.dt.to_pydatetime().dask) == set(ddf.dt_col.dt.to_pydatetime().dask)