from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
@pytest.fixture
def df_ddf():
    import numpy as np
    df = pd.DataFrame({'str_col': ['abc', 'bcd', 'cdef', 'DEFG'], 'int_col': [1, 2, 3, 4], 'dt_col': np.array([int(1000000000.0), int(1100000000.0), int(1200000000.0), None], dtype='M8[ns]')}, index=['E', 'f', 'g', 'h'])
    df['string_col'] = df['str_col'].astype('string')
    df.loc['E', 'string_col'] = pd.NA
    ddf = dd.from_pandas(df, 2)
    return (df, ddf)