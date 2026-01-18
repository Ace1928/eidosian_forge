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
def test_make_augmented_matrix():
    np.random.seed(1)
    n = 500
    x = np.random.uniform(-1, 1, (n, 3))
    s = np.dot(x.T, x)
    y = np.array(list(range(n)))
    w = np.random.uniform(0, 1, n)
    nobs, n_columns = x.shape
    alpha = 0
    aug_y, aug_x, aug_w = make_augmented_matrix(y, x, alpha * s, w)
    expected_aug_x = x
    assert_allclose(aug_x, expected_aug_x)
    expected_aug_y = y
    expected_aug_y[:nobs] = y
    assert_allclose(aug_y, expected_aug_y)
    expected_aug_w = w
    assert_allclose(aug_w, expected_aug_w)
    alpha = 1
    aug_y, aug_x, aug_w = make_augmented_matrix(y, x, s, w)
    rs = matrix_sqrt(alpha * s)
    assert_allclose(np.dot(rs.T, rs), alpha * s)
    expected_aug_x = np.vstack([x, rs])
    assert_allclose(aug_x, expected_aug_x)
    expected_aug_y = np.zeros(shape=(nobs + n_columns,))
    expected_aug_y[:nobs] = y
    assert_allclose(aug_y, expected_aug_y)
    expected_aug_w = np.concatenate((w, [1] * n_columns), axis=0)
    assert_allclose(aug_w, expected_aug_w)