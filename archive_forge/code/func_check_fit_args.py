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
def check_fit_args(distfn, arg, rvs, method):
    with np.errstate(all='ignore'), npt.suppress_warnings() as sup:
        sup.filter(category=RuntimeWarning, message='The shape parameter of the erlang')
        sup.filter(category=RuntimeWarning, message='floating point number truncated')
        vals = distfn.fit(rvs, method=method)
        vals2 = distfn.fit(rvs, optimizer='powell', method=method)
    npt.assert_(len(vals) == 2 + len(arg))
    npt.assert_(len(vals2) == 2 + len(arg))