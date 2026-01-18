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
@pytest.mark.parametrize('loc', [25.0, 10, 35])
@pytest.mark.parametrize('scale', [13, 5, 20])
def test_fit_nu(self, loc, scale):
    nu = 0.5
    n = 100
    samples = stats.nakagami.rvs(size=n, nu=nu, loc=loc, scale=scale, random_state=1337)
    nu_est, loc_est, scale_est = stats.nakagami.fit(samples, f0=nu)
    loc_theo = np.min(samples)
    scale_theo = np.sqrt(np.mean((samples - loc_est) ** 2))
    assert_allclose(nu_est, nu, rtol=1e-07)
    assert_allclose(loc_est, loc_theo, rtol=1e-07)
    assert_allclose(scale_est, scale_theo, rtol=1e-07)