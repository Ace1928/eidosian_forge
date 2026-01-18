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
def test_moments_and_entropy(self):
    a, b, c, d = (-3, -1, 2, 3)
    p1, p2, loc, scale = ((b - a) / (d - a), (c - a) / (d - a), a, d - a)
    h = 2 / (d + c - b - a)

    def moment(n):
        return h * ((d ** (n + 2) - c ** (n + 2)) / (d - c) - (b ** (n + 2) - a ** (n + 2)) / (b - a)) / (n + 1) / (n + 2)
    mean = moment(1)
    var = moment(2) - mean ** 2
    entropy = 0.5 * (d - c + b - a) / (d + c - b - a) + np.log(0.5 * (d + c - b - a))
    assert_almost_equal(stats.trapezoid.mean(p1, p2, loc, scale), mean, decimal=13)
    assert_almost_equal(stats.trapezoid.var(p1, p2, loc, scale), var, decimal=13)
    assert_almost_equal(stats.trapezoid.entropy(p1, p2, loc, scale), entropy, decimal=13)
    assert_almost_equal(stats.trapezoid.mean(0, 0, -3, 6), -1, decimal=13)
    assert_almost_equal(stats.trapezoid.mean(0, 1, -3, 6), 0, decimal=13)
    assert_almost_equal(stats.trapezoid.var(0, 1, -3, 6), 3, decimal=13)