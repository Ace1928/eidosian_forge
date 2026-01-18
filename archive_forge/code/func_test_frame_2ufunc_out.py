from __future__ import annotations
import warnings
import pytest
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
def test_frame_2ufunc_out():
    input_matrix = np.random.randint(1, 100, size=(20, 2))
    df = pd.DataFrame(input_matrix, columns=['A', 'B'])
    ddf = dd.from_pandas(df, 3)
    df_out = pd.DataFrame(np.random.randint(1, 100, size=(20, 3)), columns=['X', 'Y', 'Z'])
    ddf_out = dd.from_pandas(df_out, 3)
    with pytest.raises(ValueError):
        np.sin(ddf, out=ddf_out)
    ddf_out = dd.from_pandas(pd.Series([0]), 1)
    with pytest.raises(TypeError):
        np.sin(ddf, out=ddf_out)
    df_out = pd.DataFrame(np.random.randint(1, 100, size=(20, 2)), columns=['X', 'Y'])
    ddf_out = dd.from_pandas(df_out, 3)
    np.sin(ddf, out=ddf_out)
    np.add(ddf_out, 10, out=ddf_out)
    expected = pd.DataFrame(np.sin(input_matrix) + 10, columns=['A', 'B'])
    assert_eq(ddf_out, expected)