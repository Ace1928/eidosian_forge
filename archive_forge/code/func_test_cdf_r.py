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
def test_cdf_r(self):
    vals_R = np.array([1.01881590921421e-13, 6.13697274983578e-11, 3.37504977637992e-08, 1.55258698166181e-05, 0.00447005453832497, 0.228935323956347, 0.755759458895243, 0.953061062884484, 0.992598013917513, 0.998942646586662])
    lmbda, alpha, beta = (2, 2, 1)
    mu, delta = (0.5, 1.5)
    args = (lmbda, alpha * delta, beta * delta)
    gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
    x = np.linspace(-10, 10, 10)
    assert_allclose(gh.cdf(x), vals_R, atol=0, rtol=1e-06)