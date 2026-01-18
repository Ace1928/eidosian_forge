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
def cov_params_robust(self):
    """
        (array) The QMLE variance / covariance matrix. Computed using the
        numerical Hessian as the evaluated hessian.
        """
    cov_opg = self.cov_params_opg
    evaluated_hessian = self.model.hessian(self.params, transformed=True)
    cov_params, singular_values = pinv_extended(np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian))
    if self._rank is None:
        self._rank = np.linalg.matrix_rank(np.diag(singular_values))
    return cov_params