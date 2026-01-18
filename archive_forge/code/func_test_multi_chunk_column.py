from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
def test_multi_chunk_column() -> None:
    pytest.importorskip('pyarrow', '11.0.0')
    ser = pd.Series([1, 2, None], dtype='Int64[pyarrow]')
    df = pd.concat([ser, ser], ignore_index=True).to_frame('a')
    df_orig = df.copy()
    with pytest.raises(RuntimeError, match='Found multi-chunk pyarrow array, but `allow_copy` is False'):
        pd.api.interchange.from_dataframe(df.__dataframe__(allow_copy=False))
    result = pd.api.interchange.from_dataframe(df.__dataframe__(allow_copy=True))
    expected = pd.DataFrame({'a': [1.0, 2.0, None, 1.0, 2.0, None]}, dtype='float64')
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(df, df_orig)
    assert len(df['a'].array._pa_array.chunks) == 2
    assert len(df_orig['a'].array._pa_array.chunks) == 2