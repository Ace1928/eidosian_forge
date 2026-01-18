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
def test_fix_fit_2args_lognorm(self):
    np.random.seed(12345)
    with np.errstate(all='ignore'):
        x = stats.lognorm.rvs(0.25, 0.0, 20.0, size=20)
        expected_shape = np.sqrt(((np.log(x) - np.log(20)) ** 2).mean())
        assert_allclose(np.array(stats.lognorm.fit(x, floc=0, fscale=20)), [expected_shape, 0, 20], atol=1e-08)