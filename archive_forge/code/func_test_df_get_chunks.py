import ctypes
import math
import pytest
import modin.pandas as pd
@pytest.mark.parametrize(['size', 'n_chunks'], [(10, 3), (12, 3), (12, 5)])
def test_df_get_chunks(size, n_chunks, df_from_dict):
    df = df_from_dict({'x': list(range(size))})
    dfX = df.__dataframe__()
    chunks = list(dfX.get_chunks(n_chunks))
    assert len(chunks) == n_chunks
    assert sum((chunk.num_rows() for chunk in chunks)) == size