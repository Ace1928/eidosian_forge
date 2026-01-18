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
def test_kde_bw_positive():
    x = np.array([4.59511985, 4.59511985, 4.59511985, 4.59511985, 4.59511985, 4.59511985, 4.59511985, 4.59511985, 4.59511985, 4.59511985, 5.67332327, 6.19847872, 7.43189192])
    kde = KDE(x)
    kde.fit()
    assert kde.bw > 0