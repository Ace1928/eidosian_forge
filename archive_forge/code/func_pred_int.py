from collections import OrderedDict
import contextlib
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen
from statsmodels.base.covtype import descriptions
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import (
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import (
from statsmodels.tsa.tsatools import freq_to_period
def pred_int(self, alpha=0.05):
    """
        Calculates prediction intervals by performing multiple simulations.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the prediction interval. Default is
            0.05, that is, a 95% prediction interval.
        """
    if self.method == 'simulated':
        simulated_upper_pi = np.quantile(self.simulation_results, 1 - alpha / 2, axis=1)
        simulated_lower_pi = np.quantile(self.simulation_results, alpha / 2, axis=1)
        pred_int = np.vstack((simulated_lower_pi, simulated_upper_pi)).T
    else:
        q = norm.ppf(1 - alpha / 2)
        half_interval_size = q * np.sqrt(self.forecast_variance)
        pred_int = np.vstack((self.predicted_mean - half_interval_size, self.predicted_mean + half_interval_size)).T
    if self.use_pandas:
        pred_int = pd.DataFrame(pred_int, index=self.row_labels)
        names = [f'lower PI (alpha={alpha:f})', f'upper PI (alpha={alpha:f})']
        pred_int.columns = names
    return pred_int