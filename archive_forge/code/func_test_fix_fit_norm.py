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
def test_fix_fit_norm(self):
    x = np.arange(1, 6)
    loc, scale = stats.norm.fit(x)
    assert_almost_equal(loc, 3)
    assert_almost_equal(scale, np.sqrt(2))
    loc, scale = stats.norm.fit(x, floc=2)
    assert_equal(loc, 2)
    assert_equal(scale, np.sqrt(3))
    loc, scale = stats.norm.fit(x, fscale=2)
    assert_almost_equal(loc, 3)
    assert_equal(scale, 2)