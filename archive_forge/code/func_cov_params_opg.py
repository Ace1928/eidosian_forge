import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
@cache_readonly
def cov_params_opg(self):
    """
        (array) The variance / covariance matrix. Computed using the outer
        product of gradients method.
        """
    score_obs = self.model.score_obs(self.params, transformed=True).T
    cov_params, singular_values = pinv_extended(np.inner(score_obs, score_obs))
    if self._rank is None:
        self._rank = np.linalg.matrix_rank(np.diag(singular_values))
    return cov_params