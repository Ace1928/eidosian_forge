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
def initial_probabilities(self, params, regime_transition=None):
    """
        Retrieve initial probabilities
        """
    params = np.array(params, ndmin=1)
    if self._initialization == 'steady-state':
        if regime_transition is None:
            regime_transition = self.regime_transition_matrix(params)
        if regime_transition.ndim == 3:
            regime_transition = regime_transition[..., 0]
        m = regime_transition.shape[0]
        A = np.c_[(np.eye(m) - regime_transition).T, np.ones(m)].T
        try:
            probabilities = np.linalg.pinv(A)[:, -1]
        except np.linalg.LinAlgError:
            raise RuntimeError('Steady-state probabilities could not be constructed.')
    elif self._initialization == 'known':
        probabilities = self._initial_probabilities
    else:
        raise RuntimeError('Invalid initialization method selected.')
    probabilities = np.maximum(probabilities, 1e-20)
    return probabilities