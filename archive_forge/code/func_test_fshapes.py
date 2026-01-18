import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
@pytest.mark.slow
@pytest.mark.parametrize('method', ['MLE', 'MM'])
def test_fshapes(self, method):
    a, b = (3.0, 4.0)
    x = stats.beta.rvs(a, b, size=100, random_state=1234)
    res_1 = stats.beta.fit(x, f0=3.0, method=method)
    res_2 = stats.beta.fit(x, fa=3.0, method=method)
    assert_allclose(res_1, res_2, atol=1e-12, rtol=1e-12)
    res_2 = stats.beta.fit(x, fix_a=3.0, method=method)
    assert_allclose(res_1, res_2, atol=1e-12, rtol=1e-12)
    res_3 = stats.beta.fit(x, f1=4.0, method=method)
    res_4 = stats.beta.fit(x, fb=4.0, method=method)
    assert_allclose(res_3, res_4, atol=1e-12, rtol=1e-12)
    res_4 = stats.beta.fit(x, fix_b=4.0, method=method)
    assert_allclose(res_3, res_4, atol=1e-12, rtol=1e-12)
    assert_raises(ValueError, stats.beta.fit, x, fa=1, f0=2, method=method)
    assert_raises(ValueError, stats.beta.fit, x, fa=0, f1=1, floc=2, fscale=3, method=method)
    res_5 = stats.beta.fit(x, fa=3.0, floc=0, fscale=1, method=method)
    aa, bb, ll, ss = res_5
    assert_equal([aa, ll, ss], [3.0, 0, 1])
    a = 3.0
    data = stats.gamma.rvs(a, size=100)
    aa, ll, ss = stats.gamma.fit(data, fa=a, method=method)
    assert_equal(aa, a)