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
def partial_values(self, smooth_index, include_constant=True):
    """contribution of a smooth term to the linear prediction

        Warning: This will be replaced by a predict method

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms
        include_constant : bool
            If true, then the estimated intercept is added to the prediction
            and its standard errors. This avoids that the confidence interval
            has zero width at the imposed identification constraint, e.g.
            either at a reference point or at the mean.

        Returns
        -------
        predicted : nd_array
            predicted value of linear term.
            This is not the expected response if the link function is not
            linear.
        se_pred : nd_array
            standard error of linear prediction
        """
    variable = smooth_index
    smoother = self.model.smoother
    mask = smoother.mask[variable]
    start_idx = self.model.k_exog_linear
    idx = start_idx + np.nonzero(mask)[0]
    exog_part = smoother.basis[:, mask]
    const_idx = self.model.data.const_idx
    if include_constant and const_idx is not None:
        idx = np.concatenate(([const_idx], idx))
        exog_part = self.model.exog[:, idx]
    linpred = np.dot(exog_part, self.params[idx])
    partial_cov_params = self.cov_params(column=idx)
    covb = partial_cov_params
    var = (exog_part * np.dot(covb, exog_part.T).T).sum(1)
    se = np.sqrt(var)
    return (linpred, se)