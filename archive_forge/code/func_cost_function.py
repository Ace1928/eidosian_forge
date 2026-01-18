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
def cost_function(params, pol, y, alpha):
    lin_pred = np.dot(pol.basis, params)
    gaussian = Gaussian()
    expval = gaussian.link.inverse(lin_pred)
    loglike = gaussian.loglike(y, expval)
    itg = integral(params)
    return (loglike - alpha * itg, loglike, itg)