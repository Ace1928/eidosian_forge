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
def check_discr_samples(rng, pv, mv_ex, rtol=0.001, atol=0.1):
    rvs = rng.rvs(100000)
    mv = (rvs.mean(), rvs.var())
    assert_allclose(mv, mv_ex, rtol=rtol, atol=atol)
    pv = pv / pv.sum()
    obs_freqs = np.zeros_like(pv)
    _, freqs = np.unique(rvs, return_counts=True)
    freqs = freqs / freqs.sum()
    obs_freqs[:freqs.size] = freqs
    pval = chisquare(obs_freqs, pv).pvalue
    assert pval > 0.1