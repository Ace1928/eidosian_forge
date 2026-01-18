import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_SquareTable_nonsquare():
    tab = [[1, 0, 3], [2, 1, 4], [3, 0, 5]]
    df = pd.DataFrame(tab, index=[0, 1, 3], columns=[0, 2, 3])
    df2 = ctab.SquareTable(df, shift_zeros=False)
    e = np.asarray([[1, 0, 0, 3], [2, 0, 1, 4], [0, 0, 0, 0], [3, 0, 0, 5]], dtype=np.float64)
    assert_equal(e, df2.table)