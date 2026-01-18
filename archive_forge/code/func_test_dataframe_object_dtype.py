from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
@requires_pandas
@pytest.mark.parametrize('dtype', [object, 'string[python]'])
def test_dataframe_object_dtype(dtype):
    x = 'x' * 100000
    y = 'y' * 100000
    z = 'z' * 100000
    w = 'w' * 100000
    objs = [x, y, z, w]
    df1 = pd.DataFrame([objs * 3] * 1000, dtype=dtype)
    assert 400000 < sizeof(df1) < 550000
    df2 = pd.DataFrame([[x, y], [z, w]], dtype=dtype)
    df3 = pd.DataFrame([[x, y], [z, x]], dtype=dtype)
    df4 = pd.DataFrame([[x, x], [x, x]], dtype=dtype)
    assert sizeof(df4) < sizeof(df3) < sizeof(df2)