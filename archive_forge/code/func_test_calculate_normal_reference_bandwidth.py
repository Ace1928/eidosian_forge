import numpy as np
from scipy import stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.bandwidths import select_bandwidth
from statsmodels.nonparametric.bandwidths import bw_normal_reference
from numpy.testing import assert_allclose
import pytest
def test_calculate_normal_reference_bandwidth(self):
    bw_expected = 0.2978114711369889
    bw = bw_normal_reference(Xi)
    assert_allclose(bw, bw_expected)