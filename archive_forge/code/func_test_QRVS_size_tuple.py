import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
def test_QRVS_size_tuple(self, method):
    dist = StandardNormal()
    Method = getattr(stats.sampling, method)
    gen = Method(dist)
    size = (3, 4)
    d = 5
    qrng = stats.qmc.Halton(d, seed=0)
    qrng2 = stats.qmc.Halton(d, seed=0)
    uniform = qrng2.random(np.prod(size))
    qrvs = gen.qrvs(size=size, d=d, qmc_engine=qrng)
    qrvs2 = stats.norm.ppf(uniform)
    for i in range(d):
        sample = qrvs[..., i]
        sample2 = qrvs2[:, i].reshape(size)
        assert_allclose(sample, sample2, atol=1e-12)