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
@property
def expected_durations(self):
    """
        (array) Expected duration of a regime, possibly time-varying.
        """
    diag = np.diagonal(self.regime_transition)
    expected_durations = np.zeros_like(diag)
    degenerate = np.any(diag == 1, axis=1)
    expected_durations[~degenerate] = 1 / (1 - diag[~degenerate])
    expected_durations[degenerate] = np.nan
    expected_durations[diag == 1] = np.inf
    return expected_durations.squeeze()