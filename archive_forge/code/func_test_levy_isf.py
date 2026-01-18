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
@pytest.mark.parametrize('p, expected_isf', [(1e-20, 6.366197723675814e+39), (1e-08, 6366197723675813.0), (0.375, 4.185810119346273), (0.875, 0.42489442055310134), (0.999, 0.09235685880262713), (0.9999999962747097, 0.028766845244146945)])
def test_levy_isf(p, expected_isf):
    x = stats.levy.isf(p)
    assert_allclose(x, expected_isf, atol=5e-15)