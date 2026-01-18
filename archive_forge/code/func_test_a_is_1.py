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
def test_a_is_1(self):
    x = np.logspace(-4, -1, 4)
    a = 1
    c = 100
    p = stats.exponweib.pdf(x, a, c)
    expected = stats.weibull_min.pdf(x, c)
    assert_allclose(p, expected)
    logp = stats.exponweib.logpdf(x, a, c)
    expected = stats.weibull_min.logpdf(x, c)
    assert_allclose(logp, expected)