from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
def compute_conf_interval_95(mu, alpha, p, prob_infl, nobs):
    dispersion_factor = 1 + alpha * mu ** (p - 1) + prob_infl * mu
    var = (dispersion_factor * (1 - prob_infl) * mu).mean()
    var += (((1 - prob_infl) * mu) ** 2).mean()
    var -= ((1 - prob_infl) * mu).mean() ** 2
    std = np.sqrt(var)
    conf_intv_95 = 2 * std / np.sqrt(nobs)
    return conf_intv_95