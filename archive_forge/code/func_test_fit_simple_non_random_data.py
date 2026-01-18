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
def test_fit_simple_non_random_data(self):
    data = np.array([1.0, 1.0, 3.0, 5.0, 8.0, 14.0])
    loc, scale = stats.laplace.fit(data, floc=6)
    assert_allclose(scale, 4, atol=1e-15, rtol=1e-15)
    loc, scale = stats.laplace.fit(data, fscale=6)
    assert_allclose(loc, 4, atol=1e-15, rtol=1e-15)