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
@pytest.mark.parametrize('dist,args', distcont)
def test_fit_w_non_finite_data_values(self, dist, args):
    """gh-10300"""
    if dist in self.fitSkipNonFinite:
        pytest.skip('%s fit known to fail or deprecated' % dist)
    x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
    y = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
    distfunc = getattr(stats, dist)
    assert_raises(ValueError, distfunc.fit, x, fscale=1)
    assert_raises(ValueError, distfunc.fit, y, fscale=1)