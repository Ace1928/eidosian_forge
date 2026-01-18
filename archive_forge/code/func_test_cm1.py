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
def test_cm1(self):
    rv = stats.genpareto(c=-1.0)
    x = np.linspace(0, 10.0, 30)
    assert_allclose(rv.pdf(x), stats.uniform.pdf(x))
    assert_allclose(rv.cdf(x), stats.uniform.cdf(x))
    assert_allclose(rv.sf(x), stats.uniform.sf(x))
    q = np.linspace(0.0, 1.0, 10)
    assert_allclose(rv.ppf(q), stats.uniform.ppf(q))
    assert_allclose(rv.logpdf(1), 0)