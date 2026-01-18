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
def test_check_is_fit_ok_with_standard_custom_bandwidth(self):
    kde = self.kde.fit(bw=bandwidths.bw_silverman)
    s1 = kde.support.copy()
    d1 = kde.density.copy()
    kde = self.kde.fit(bw='silverman')
    npt.assert_almost_equal(s1, kde.support, self.decimal_density)
    npt.assert_almost_equal(d1, kde.density, self.decimal_density)