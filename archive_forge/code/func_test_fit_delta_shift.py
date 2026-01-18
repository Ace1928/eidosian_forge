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
def test_fit_delta_shift(self):
    SHIFT = 1
    x = np.array([1, 1, 3, 3, 10, 10, 10, 30, 30, 100, 100])
    alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(-x)
    alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(-x + SHIFT)
    assert_almost_equal(alpha2, alpha1)
    assert_almost_equal(beta2, beta1)
    assert_almost_equal(loc2, loc1 + SHIFT)
    assert_almost_equal(scale2, scale1)