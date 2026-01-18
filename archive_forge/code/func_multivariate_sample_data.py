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
def multivariate_sample_data(seed=1):
    n = 1000
    x1 = np.linspace(-1, 1, n)
    x2 = np.linspace(-10, 10, n)
    x = np.vstack([x1, x2]).T
    np.random.seed(seed)
    y = x1 * x1 * x1 + x2 + np.random.normal(0, 0.01, n)
    degree1 = 4
    degree2 = 3
    degrees = [degree1, degree2]
    pol = PolynomialSmoother(x, degrees)
    return (x, y, pol)