import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import block_diag
import pytest
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.gam.smooth_basis import (
from statsmodels.gam.generalized_additive_model import (
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_penalties import (UnivariateGamPenalty,
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Gaussian
from statsmodels.genmod.generalized_linear_model import lm
def test_glm_pirls_compatibility():
    np.random.seed(0)
    n = 500
    x1 = np.linspace(-3, 3, n)
    x2 = np.random.rand(n)
    x = np.vstack([x1, x2]).T
    y1 = np.sin(x1) / x1
    y2 = x2 * x2
    y0 = y1 + y2
    y = y0 + np.random.normal(0, 0.3, n)
    y -= y.mean()
    y0 -= y0.mean()
    alphas = [5.75] * 2
    alphas_glm = [1.2] * 2
    cs = BSplines(x, df=[10, 10], degree=[3, 3], constraints='center')
    gam_pirls = GLMGam(y, smoother=cs, alpha=alphas)
    gam_glm = GLMGam(y, smoother=cs, alpha=alphas)
    gam_res_glm = gam_glm.fit(method='nm', max_start_irls=0, disp=1, maxiter=20000)
    gam_res_glm = gam_glm.fit(start_params=gam_res_glm.params, method='bfgs', max_start_irls=0, disp=1, maxiter=20000)
    gam_res_pirls = gam_pirls.fit()
    y_est_glm = np.dot(cs.basis, gam_res_glm.params)
    y_est_glm -= y_est_glm.mean()
    y_est_pirls = np.dot(cs.basis, gam_res_pirls.params)
    y_est_pirls -= y_est_pirls.mean()
    assert_allclose(gam_res_glm.params, gam_res_pirls.params, atol=5e-05, rtol=5e-05)
    assert_allclose(y_est_glm, y_est_pirls, atol=5e-05)