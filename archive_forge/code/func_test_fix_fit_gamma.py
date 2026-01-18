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
def test_fix_fit_gamma(self):
    x = np.arange(1, 6)
    meanlog = np.log(x).mean()
    floc = 0
    a, loc, scale = stats.gamma.fit(x, floc=floc)
    s = np.log(x.mean()) - meanlog
    assert_almost_equal(np.log(a) - special.digamma(a), s, decimal=5)
    assert_equal(loc, floc)
    assert_almost_equal(scale, x.mean() / a, decimal=8)
    f0 = 1
    floc = 0
    a, loc, scale = stats.gamma.fit(x, f0=f0, floc=floc)
    assert_equal(a, f0)
    assert_equal(loc, floc)
    assert_almost_equal(scale, x.mean() / a, decimal=8)
    f0 = 2
    floc = 0
    a, loc, scale = stats.gamma.fit(x, f0=f0, floc=floc)
    assert_equal(a, f0)
    assert_equal(loc, floc)
    assert_almost_equal(scale, x.mean() / a, decimal=8)
    floc = 0
    fscale = 2
    a, loc, scale = stats.gamma.fit(x, floc=floc, fscale=fscale)
    assert_equal(loc, floc)
    assert_equal(scale, fscale)
    c = meanlog - np.log(fscale)
    assert_almost_equal(special.digamma(a), c)