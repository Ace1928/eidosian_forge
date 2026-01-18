from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import (
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning
def compare_waldres(res, wa, constrasts):
    for i, c in enumerate(constrasts):
        wt = res.wald_test(c, scalar=True)
        assert_allclose(wa.table.values[i, 0], wt.statistic)
        assert_allclose(wa.table.values[i, 1], wt.pvalue)
        df = c.shape[0] if c.ndim == 2 else 1
        assert_equal(wa.table.values[i, 2], df)
        assert_allclose(wa.statistic[i], wt.statistic)
        assert_allclose(wa.pvalues[i], wt.pvalue)
        assert_equal(wa.df_constraints[i], df)
        if res.use_t:
            assert_equal(wa.df_denom[i], res.df_resid)
    col_names = wa.col_names
    if res.use_t:
        assert_equal(wa.distribution, 'F')
        assert_equal(col_names[0], 'F')
        assert_equal(col_names[1], 'P>F')
    else:
        assert_equal(wa.distribution, 'chi2')
        assert_equal(col_names[0], 'chi2')
        assert_equal(col_names[1], 'P>chi2')
    wa.summary_frame()