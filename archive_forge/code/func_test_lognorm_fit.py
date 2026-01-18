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
def test_lognorm_fit(self):
    x = np.array([1.5, 3, 10, 15, 23, 59])
    lnxm1 = np.log(x - 1)
    shape, loc, scale = stats.lognorm.fit(x, floc=1)
    assert_allclose(shape, lnxm1.std(), rtol=1e-12)
    assert_equal(loc, 1)
    assert_allclose(scale, np.exp(lnxm1.mean()), rtol=1e-12)
    shape, loc, scale = stats.lognorm.fit(x, floc=1, fscale=6)
    assert_allclose(shape, np.sqrt(((lnxm1 - np.log(6)) ** 2).mean()), rtol=1e-12)
    assert_equal(loc, 1)
    assert_equal(scale, 6)
    shape, loc, scale = stats.lognorm.fit(x, floc=1, fix_s=0.75)
    assert_equal(shape, 0.75)
    assert_equal(loc, 1)
    assert_allclose(scale, np.exp(lnxm1.mean()), rtol=1e-12)