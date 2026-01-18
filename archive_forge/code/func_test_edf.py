import os
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pandas as pd
import pytest
import patsy
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen
from statsmodels.gam.smooth_basis import (BSplines, CyclicCubicSplines)
from statsmodels.gam.generalized_additive_model import (
from statsmodels.tools.linalg import matrix_sqrt, transf_constraints
from .results import results_pls, results_mpg_bs, results_mpg_bs_poisson
def test_edf(self):
    res1 = self.res1
    res2 = self.res2
    assert_allclose(res1.edf, res2.edf_all, rtol=1e-06)
    hat = res1.get_hat_matrix_diag()
    assert_allclose(hat, res2.hat, rtol=1e-06)
    assert_allclose(res1.aic, res2.aic, rtol=1e-08)
    assert_allclose(res1.deviance, res2.deviance, rtol=1e-08)
    assert_allclose(res1.df_resid, res2.residual_df, rtol=1e-08)