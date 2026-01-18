from statsmodels.compat.python import lrange, lmap
import os
import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm
def test_hypothesis(self):
    res1, res2 = (self.res1, self.res2)
    restriction = np.eye(len(res1.params))
    res_t = res1.t_test(restriction)
    assert_allclose(res_t.tvalue, res1.tvalues, rtol=1e-12, atol=0)
    assert_allclose(res_t.pvalue, res1.pvalues, rtol=1e-12, atol=0)
    res_f = res1.f_test(restriction[:-1])
    assert_allclose(res_f.fvalue, res1.fvalue, rtol=1e-12, atol=0)
    assert_allclose(res_f.pvalue, res1.f_pvalue, rtol=1e-10, atol=0)
    assert_allclose(res_f.fvalue, res2.F, rtol=1e-10, atol=0)
    assert_allclose(res_f.pvalue, res2.Fp, rtol=1e-08, atol=0)