import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
def check_ppf_isf(distfn, arg, msg):
    p = np.array([0.1, 0.9])
    npt.assert_almost_equal(distfn.isf(p, *arg), distfn.ppf(1 - p, *arg), decimal=DECIMAL, err_msg=msg + ' - ppf-isf relationship')