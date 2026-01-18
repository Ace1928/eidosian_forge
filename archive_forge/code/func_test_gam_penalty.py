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
def test_gam_penalty():
    """
    test the func method of the gam penalty
    :return:
    """
    pol, y = polynomial_sample_data()
    univ_pol = pol.smoothers[0]
    alpha = 1
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=univ_pol)
    for _ in range(10):
        params = np.random.randint(-2, 2, 4)
        gp_score = gp.func(params)
        itg = integral(params)
        assert_allclose(gp_score, itg, atol=0.1)