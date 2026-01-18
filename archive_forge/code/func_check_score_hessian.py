import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
def check_score_hessian(results):
    params = results.params
    sc = results.model.score(params * 0.98, scale=1)
    llfunc = lambda x: results.model.loglike(x, scale=1)
    sc2 = approx_fprime(params * 0.98, llfunc)
    assert_allclose(sc, sc2, rtol=0.0001, atol=0.0001)
    hess = results.model.hessian(params, scale=1)
    hess2 = approx_hess(params, llfunc)
    assert_allclose(hess, hess2, rtol=0.0001)
    scfunc = lambda x: results.model.score(x, scale=1)
    hess3 = approx_fprime(params, scfunc)
    assert_allclose(hess, hess3, rtol=0.0001)