import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_mcnemar():
    b1 = ctab.mcnemar(tables[0], exact=False, correction=False)
    st = sm.stats.SquareTable(tables[0])
    b2 = st.homogeneity()
    assert_allclose(b1.statistic, b2.statistic)
    assert_equal(b2.df, 1)
    b3 = ctab.mcnemar(tables[0], exact=False, correction=True)
    assert_allclose(b3.pvalue, r_results.loc[0, 'homog_cont_p'])
    b4 = ctab.mcnemar(tables[0], exact=True)
    assert_allclose(b4.pvalue, r_results.loc[0, 'homog_binom_p'])