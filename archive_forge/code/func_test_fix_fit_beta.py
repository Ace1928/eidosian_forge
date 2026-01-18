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
def test_fix_fit_beta(self):

    def mlefunc(a, b, x):
        n = len(x)
        s1 = np.log(x).sum()
        s2 = np.log(1 - x).sum()
        psiab = special.psi(a + b)
        func = [s1 - n * (-psiab + special.psi(a)), s2 - n * (-psiab + special.psi(b))]
        return func
    x = np.array([0.125, 0.25, 0.5])
    a, b, loc, scale = stats.beta.fit(x, floc=0, fscale=1)
    assert_equal(loc, 0)
    assert_equal(scale, 1)
    assert_allclose(mlefunc(a, b, x), [0, 0], atol=1e-06)
    x = np.array([0.125, 0.25, 0.5])
    a, b, loc, scale = stats.beta.fit(x, f0=2, floc=0, fscale=1)
    assert_equal(a, 2)
    assert_equal(loc, 0)
    assert_equal(scale, 1)
    da, db = mlefunc(a, b, x)
    assert_allclose(db, 0, atol=1e-05)
    x2 = 1 - x
    a2, b2, loc2, scale2 = stats.beta.fit(x2, f1=2, floc=0, fscale=1)
    assert_equal(b2, 2)
    assert_equal(loc2, 0)
    assert_equal(scale2, 1)
    da, db = mlefunc(a2, b2, x2)
    assert_allclose(da, 0, atol=1e-05)
    assert_almost_equal(a2, b)
    assert_raises(ValueError, stats.beta.fit, x, floc=0.5, fscale=1)
    y = np.array([0, 0.5, 1])
    assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1)
    assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1, f0=2)
    assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1, f1=2)
    assert_raises(ValueError, stats.beta.fit, y, f0=0, f1=1, floc=2, fscale=3)