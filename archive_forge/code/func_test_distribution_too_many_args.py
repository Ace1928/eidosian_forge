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
def test_distribution_too_many_args():
    np.random.seed(1234)
    x = np.linspace(0.1, 0.7, num=5)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, loc=1.0)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, 4, loc=1.0)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, 4, 5)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.rvs, 2.0, 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.cdf, x, 2.0, 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.ppf, x, 2.0, 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.stats, 2.0, 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.entropy, 2.0, 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.fit, x, 2.0, 3, loc=1.0, scale=0.5)
    stats.gamma.pdf(x, 2, 3)
    stats.gamma.pdf(x, 2, 3, 4)
    stats.gamma.stats(2.0, 3)
    stats.gamma.stats(2.0, 3, 4)
    stats.gamma.stats(2.0, 3, 4, 'mv')
    stats.gamma.rvs(2.0, 3, 4, 5)
    stats.gamma.fit(stats.gamma.rvs(2.0, size=7), 2.0)
    stats.geom.pmf(x, 2, loc=3)
    assert_raises(TypeError, stats.geom.pmf, x, 2, 3, 4)
    assert_raises(TypeError, stats.geom.pmf, x, 2, 3, loc=4)
    assert_raises(TypeError, stats.expon.pdf, x, 3, loc=1.0)
    assert_raises(TypeError, stats.exponweib.pdf, x, 3, 4, 5, loc=1.0)
    assert_raises(TypeError, stats.exponweib.pdf, x, 3, 4, 5, 0.1, 0.1)
    assert_raises(TypeError, stats.ncf.pdf, x, 3, 4, 5, 6, loc=1.0)
    assert_raises(TypeError, stats.ncf.pdf, x, 3, 4, 5, 6, 1.0, scale=0.5)
    stats.ncf.pdf(x, 3, 4, 5, 6, 1.0)