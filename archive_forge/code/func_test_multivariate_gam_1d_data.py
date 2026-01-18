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
def test_multivariate_gam_1d_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, 'results', 'prediction_from_mgcv.csv')
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.y
    df = [10]
    degree = [3]
    bsplines = BSplines(x, degree=degree, df=df)
    y_mgcv = data_from_r.y_est
    alpha = [0.0168 * 0.0251 / 2 * 500]
    gp = MultivariateGamPenalty(bsplines, alpha=alpha)
    glm_gam = GLMGam(y, exog=np.ones((len(y), 1)), smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method='pirls', max_start_irls=0, disp=1, maxiter=10000)
    y_gam = res_glm_gam.fittedvalues
    assert_allclose(y_gam, y_mgcv, atol=0.01)