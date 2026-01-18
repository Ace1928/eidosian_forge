import numpy as np
from scipy import stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.bandwidths import select_bandwidth
from statsmodels.nonparametric.bandwidths import bw_normal_reference
from numpy.testing import assert_allclose
import pytest
def test_calculate_bandwidth_gaussian(self):
    bw_expected = [0.29774853596742024, 0.2530440815587141, 0.2978114711369889]
    kern = kernels.Gaussian()
    bw_calc = [0, 0, 0]
    for ii, bw in enumerate(['scott', 'silverman', 'normal_reference']):
        bw_calc[ii] = select_bandwidth(Xi, bw, kern)
    assert_allclose(bw_expected, bw_calc)