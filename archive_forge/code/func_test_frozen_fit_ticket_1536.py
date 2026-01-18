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
def test_frozen_fit_ticket_1536():
    np.random.seed(5678)
    true = np.array([0.25, 0.0, 0.5])
    x = stats.lognorm.rvs(true[0], true[1], true[2], size=100)
    with np.errstate(divide='ignore'):
        params = np.array(stats.lognorm.fit(x, floc=0.0))
    assert_almost_equal(params, true, decimal=2)
    params = np.array(stats.lognorm.fit(x, fscale=0.5, loc=0))
    assert_almost_equal(params, true, decimal=2)
    params = np.array(stats.lognorm.fit(x, f0=0.25, loc=0))
    assert_almost_equal(params, true, decimal=2)
    params = np.array(stats.lognorm.fit(x, f0=0.25, floc=0))
    assert_almost_equal(params, true, decimal=2)
    np.random.seed(5678)
    loc = 1
    floc = 0.9
    x = stats.norm.rvs(loc, 2.0, size=100)
    params = np.array(stats.norm.fit(x, floc=floc))
    expected = np.array([floc, np.sqrt(((x - floc) ** 2).mean())])
    assert_almost_equal(params, expected, decimal=4)