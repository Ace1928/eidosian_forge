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
@pytest.mark.parametrize('testlogcdf', [True, False])
def test_logcdfsf_tails(self, testlogcdf):
    x = np.array([-10000, -800, 17, 50, 500])
    if testlogcdf:
        y = stats.logistic.logcdf(x)
    else:
        y = stats.logistic.logsf(-x)
    expected = [-10000.0, -800.0, -4.139937633089748e-08, -1.9287498479639178e-22, -7.124576406741286e-218]
    assert_allclose(y, expected, rtol=2e-15)