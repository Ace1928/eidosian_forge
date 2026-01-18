import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
def check_moment_frozen(distfn, arg, m, k):
    npt.assert_allclose(distfn(*arg).moment(k), m, atol=1e-10, rtol=1e-10)