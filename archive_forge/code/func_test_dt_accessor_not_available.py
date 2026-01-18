from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
def test_dt_accessor_not_available(df_ddf):
    df, ddf = df_ddf
    with pytest.raises(AttributeError) as exc:
        ddf.str_col.dt
    assert '.dt accessor' in str(exc.value)