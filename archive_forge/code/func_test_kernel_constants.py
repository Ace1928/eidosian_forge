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
def test_kernel_constants(self):
    kern = self.res1.kernel
    nc = kern.norm_const
    kern._norm_const = None
    nc2 = kern.norm_const
    npt.assert_allclose(nc, nc2, rtol=1e-10)
    l2n = kern.L2Norm
    kern._L2Norm = None
    l2n2 = kern.L2Norm
    npt.assert_allclose(l2n, l2n2, rtol=1e-10)
    v = kern.kernel_var
    kern._kernel_var = None
    v2 = kern.kernel_var
    npt.assert_allclose(v, v2, rtol=1e-10)