import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_SquareTable_from_data():
    np.random.seed(434)
    df = pd.DataFrame(index=range(100), columns=['v1', 'v2'])
    df['v1'] = np.random.randint(0, 5, 100)
    df['v2'] = np.random.randint(0, 5, 100)
    table = pd.crosstab(df['v1'], df['v2'])
    rslt1 = ctab.SquareTable(table)
    rslt2 = ctab.SquareTable.from_data(df)
    rslt3 = ctab.SquareTable(np.asarray(table))
    assert_equal(rslt1.summary().as_text(), rslt2.summary().as_text())
    assert_equal(rslt2.summary().as_text(), rslt3.summary().as_text())
    s = str(rslt1)
    assert_equal(s.startswith('A 5x5 contingency table with counts:'), True)
    assert_equal(rslt1.table[0, 0], 8.0)