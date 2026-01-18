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
def predicted_marginal_probabilities(self):
    if self._predicted_marginal_probabilities is None:
        self._predicted_marginal_probabilities = self.predicted_joint_probabilities
        for i in range(self._predicted_marginal_probabilities.ndim - 2):
            self._predicted_marginal_probabilities = np.sum(self._predicted_marginal_probabilities, axis=-2)
    return self._predicted_marginal_probabilities