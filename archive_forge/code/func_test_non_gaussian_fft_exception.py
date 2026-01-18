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
def test_non_gaussian_fft_exception(self):
    with pytest.raises(NotImplementedError):
        self.kde.fit(kernel='epa', gridsize=50, fft=True, bw='silverman')