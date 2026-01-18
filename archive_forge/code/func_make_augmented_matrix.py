from collections.abc import Iterable
import copy  # check if needed when dropping python 2.7
import numpy as np
from scipy import optimize
import pandas as pd
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import (
import statsmodels.regression.linear_model as lm
from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
def make_augmented_matrix(endog, exog, penalty_matrix, weights):
    """augment endog, exog and weights with stochastic restriction matrix

    Parameters
    ----------
    endog : ndarray
        response or endogenous variable
    exog : ndarray
        design matrix, matrix of exogenous or explanatory variables
    penalty_matrix : ndarray, 2-Dim square
        penality matrix for quadratic penalization
    weights : ndarray
        weights for WLS

    Returns
    -------
    endog_aug : ndarray
        augmented response variable
    exog_aug : ndarray
        augmented design matrix
    weights_aug : ndarray
        augmented weights for WLS
    """
    y, x, s = (endog, exog, penalty_matrix)
    nobs = x.shape[0]
    rs = matrix_sqrt(s)
    x1 = np.vstack([x, rs])
    n_samp1es_x1 = x1.shape[0]
    y1 = np.array([0.0] * n_samp1es_x1)
    y1[:nobs] = y
    id1 = np.array([1.0] * rs.shape[0])
    w1 = np.concatenate([weights, id1])
    return (y1, x1, w1)