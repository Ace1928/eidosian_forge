import numpy as np
import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def test_nan_stays_float():
    idx0 = MultiIndex(levels=[['A', 'B'], []], codes=[[1, 0], [-1, -1]], names=[0, 1])
    idx1 = MultiIndex(levels=[['C'], ['D']], codes=[[0], [0]], names=[0, 1])
    idxm = idx0.join(idx1, how='outer')
    assert pd.isna(idx0.get_level_values(1)).all()
    assert pd.isna(idxm.get_level_values(1)[:-1]).all()
    df0 = pd.DataFrame([[1, 2]], index=idx0)
    df1 = pd.DataFrame([[3, 4]], index=idx1)
    dfm = df0 - df1
    assert pd.isna(df0.index.get_level_values(1)).all()
    assert pd.isna(dfm.index.get_level_values(1)[:-1]).all()