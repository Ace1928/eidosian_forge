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
def test_beta(self):
    v = stats.beta.expect(lambda x: (x - 19 / 3.0) * (x - 19 / 3.0), args=(10, 5), loc=5, scale=2)
    assert_almost_equal(v, 1.0 / 18.0, decimal=13)
    m = stats.beta.expect(lambda x: x, args=(10, 5), loc=5.0, scale=2.0)
    assert_almost_equal(m, 19 / 3.0, decimal=13)
    ub = stats.beta.ppf(0.95, 10, 10, loc=5, scale=2)
    lb = stats.beta.ppf(0.05, 10, 10, loc=5, scale=2)
    prob90 = stats.beta.expect(lambda x: 1.0, args=(10, 10), loc=5.0, scale=2.0, lb=lb, ub=ub, conditional=False)
    assert_almost_equal(prob90, 0.9, decimal=13)
    prob90c = stats.beta.expect(lambda x: 1, args=(10, 10), loc=5, scale=2, lb=lb, ub=ub, conditional=True)
    assert_almost_equal(prob90c, 1.0, decimal=13)