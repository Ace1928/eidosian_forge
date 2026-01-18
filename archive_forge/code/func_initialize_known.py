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
def initialize_known(self, probabilities, tol=1e-08):
    """
        Set initialization of regime probabilities to use known values
        """
    self._initialization = 'known'
    probabilities = np.array(probabilities, ndmin=1)
    if not probabilities.shape == (self.k_regimes,):
        raise ValueError('Initial probabilities must be a vector of shape (k_regimes,).')
    if not np.abs(np.sum(probabilities) - 1) < tol:
        raise ValueError('Initial probabilities vector must sum to one.')
    self._initial_probabilities = probabilities