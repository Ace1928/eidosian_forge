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
@pytest.mark.parametrize('chi, expected', [(0.5, (0.0142339473104779, 0.3383434069831524, 0.9120392960157007)), (0.2, (0.014844494764049919, 0.34853337610145363, 0.916373099762534)), (0.1, (0.014932902500433911, 0.34999386894914264, 0.9169794935931616)), (0.01, (0.014962141754813293, 0.35047607577486417, 0.9171789075514756)), (0.001, (0.01496243430933372, 0.35048089844774266, 0.917180899947689)), (0.0001, (0.014962437234895118, 0.3504809466745317, 0.9171809198714769)), (1e-06, (0.01496243726444329, 0.3504809471616223, 0.9171809200727071)), (1e-09, (0.014962437264446245, 0.350480947161671, 0.9171809200727272))])
def test_cdf_small_chi(self, chi, expected):
    x = np.array([0.1, 0.5, 0.9])
    assert_allclose(stats.argus.cdf(x, chi), expected, rtol=1e-12)