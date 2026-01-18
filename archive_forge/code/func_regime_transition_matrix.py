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
def regime_transition_matrix(self, params, exog_tvtp=None):
    """
        Construct the left-stochastic transition matrix

        Notes
        -----
        This matrix will either be shaped (k_regimes, k_regimes, 1) or if there
        are time-varying transition probabilities, it will be shaped
        (k_regimes, k_regimes, nobs).

        The (i,j)th element of this matrix is the probability of transitioning
        from regime j to regime i; thus the previous regime is represented in a
        column and the next regime is represented by a row.

        It is left-stochastic, meaning that each column sums to one (because
        it is certain that from one regime (j) you will transition to *some
        other regime*).
        """
    params = np.array(params, ndmin=1)
    if not self.tvtp:
        regime_transition_matrix = np.zeros((self.k_regimes, self.k_regimes, 1), dtype=np.promote_types(np.float64, params.dtype))
        regime_transition_matrix[:-1, :, 0] = np.reshape(params[self.parameters['regime_transition']], (self.k_regimes - 1, self.k_regimes))
        regime_transition_matrix[-1, :, 0] = 1 - np.sum(regime_transition_matrix[:-1, :, 0], axis=0)
    else:
        regime_transition_matrix = self._regime_transition_matrix_tvtp(params, exog_tvtp)
    return regime_transition_matrix