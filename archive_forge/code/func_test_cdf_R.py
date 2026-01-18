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
def test_cdf_R(self):
    r_cdf = np.array([8.034920282e-07, 2.512671945e-05, 0.3186661051, 0.9988650664, 0.9999848769])
    x_test = np.array([-7, -5, 0, 8, 15])
    vals_cdf = stats.norminvgauss.cdf(x_test, a=1, b=0.5)
    assert_allclose(vals_cdf, r_cdf, atol=1e-09)