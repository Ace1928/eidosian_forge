import os
import numpy.testing as npt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import statsmodels.sandbox.nonparametric.kernels as kernels
import statsmodels.nonparametric.bandwidths as bandwidths
def test_fit_self(reset_randomstate):
    x = np.random.standard_normal(100)
    kde = KDE(x)
    assert isinstance(kde, KDE)
    assert isinstance(kde.fit(), KDE)