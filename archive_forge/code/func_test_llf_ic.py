from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
import statsmodels.datasets.macrodata
from statsmodels.tsa.vector_ar.svar_model import SVAR
def test_llf_ic(self):
    res1 = self.res1
    res2 = self.res2
    assert_allclose(res1.llf, res2.ll_var, atol=1e-12)
    corr_const = -8.51363119922803
    assert_allclose(res1.fpe, res2.fpe_var, atol=1e-12)
    assert_allclose(res1.aic - corr_const, res2.aic_var, atol=1e-12)
    assert_allclose(res1.bic - corr_const, res2.sbic_var, atol=1e-12)
    assert_allclose(res1.hqic - corr_const, res2.hqic_var, atol=1e-12)