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
def test_cyclic_cubic_splines():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, 'results', 'cubic_cyclic_splines_from_mgcv.csv')
    data_from_r = pd.read_csv(file_path)
    x = data_from_r[['x0', 'x2']].values
    y = data_from_r['y'].values
    y_est_mgcv = data_from_r[['y_est']].values
    s_mgcv = data_from_r[['s(x0)', 's(x2)']].values
    dfs = [10, 10]
    ccs = CyclicCubicSplines(x, df=dfs)
    alpha = [0.05 / 2, 0.0005 / 2]
    gam = GLMGam(y, smoother=ccs, alpha=alpha)
    gam_res = gam.fit(method='pirls')
    s0 = np.dot(ccs.basis[:, ccs.mask[0]], gam_res.params[ccs.mask[0]])
    s0 -= s0.mean()
    s1 = np.dot(ccs.basis[:, ccs.mask[1]], gam_res.params[ccs.mask[1]])
    s1 -= s1.mean()
    assert_allclose(s0, s_mgcv[:, 0], atol=0.02)
    assert_allclose(s1, s_mgcv[:, 1], atol=0.33)