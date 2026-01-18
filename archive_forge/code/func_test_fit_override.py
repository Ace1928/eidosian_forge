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
@pytest.mark.parametrize('method', ['mle', 'mm'])
def test_fit_override(self, method):
    rng = np.random.default_rng(98643218961)
    rvs = stats.loguniform.rvs(0.1, 1, size=1000, random_state=rng)
    a, b, loc, scale = stats.loguniform.fit(rvs, method=method)
    assert scale == 1
    a, b, loc, scale = stats.loguniform.fit(rvs, fscale=2, method=method)
    assert scale == 2