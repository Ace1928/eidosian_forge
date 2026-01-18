from io import StringIO
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import patsy
import pytest
from statsmodels import datasets
from statsmodels.base._constraints import fit_constrained
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from .results import (
def test_basic_method(self):
    if hasattr(self, 'res1m'):
        res1 = self.res1m if not hasattr(self.res1m, '_results') else self.res1m._results
        res2 = self.res2
        assert_allclose(res1.params, res2.params[self.idx], rtol=1e-06)
        mask = (res1.bse == 0) & np.isnan(res2.bse[self.idx])
        assert_allclose(res1.bse[~mask], res2.bse[self.idx][~mask], rtol=1e-06)
        tvalues = res2.params_table[self.idx, 2]
        mask = np.isinf(res1.tvalues) & np.isnan(tvalues)
        assert_allclose(res1.tvalues[~mask], tvalues[~mask], rtol=1e-06)
        pvalues = res2.params_table[self.idx, 3]
        mask = (res1.pvalues == 0) & np.isnan(pvalues)
        assert_allclose(res1.pvalues[~mask], pvalues[~mask], rtol=5e-05)
        ci_low = res2.params_table[self.idx, 4]
        ci_upp = res2.params_table[self.idx, 5]
        ci = np.column_stack((ci_low, ci_upp))
        assert_allclose(res1.conf_int()[~np.isnan(ci)], ci[~np.isnan(ci)], rtol=5e-05)
        assert_allclose(res1.llf, res2.ll, rtol=1e-06)
        assert_equal(res1.df_model, res2.df_m)
        df_r = res2.N - res2.df_m - 1
        assert_equal(res1.df_resid, df_r)
    else:
        pytest.skip('not available yet')