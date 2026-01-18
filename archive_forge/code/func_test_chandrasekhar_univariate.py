import numpy as np
import pandas as pd
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.kalman_filter import (
from numpy.testing import assert_allclose
import pytest
def test_chandrasekhar_univariate():
    check_univariate_chandrasekhar(filter_univariate=True)
    check_univariate_chandrasekhar(filter_univariate=True, concentrate_scale=True)
    check_multivariate_chandrasekhar(filter_univariate=True)
    check_multivariate_chandrasekhar(filter_univariate=True, measurement_error=True)
    check_multivariate_chandrasekhar(filter_univariate=True, error_cov_type='diagonal')
    check_multivariate_chandrasekhar(filter_univariate=True, gen_obs_cov=True)
    check_multivariate_chandrasekhar(filter_univariate=True, gen_obs_cov=True, memory_conserve=True)