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
def test_nan_arguments_gh_issue_1362():
    with np.errstate(invalid='ignore'):
        assert_(np.isnan(stats.t.logcdf(1, np.nan)))
        assert_(np.isnan(stats.t.cdf(1, np.nan)))
        assert_(np.isnan(stats.t.logsf(1, np.nan)))
        assert_(np.isnan(stats.t.sf(1, np.nan)))
        assert_(np.isnan(stats.t.pdf(1, np.nan)))
        assert_(np.isnan(stats.t.logpdf(1, np.nan)))
        assert_(np.isnan(stats.t.ppf(1, np.nan)))
        assert_(np.isnan(stats.t.isf(1, np.nan)))
        assert_(np.isnan(stats.bernoulli.logcdf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.cdf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.logsf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.sf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.pmf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.logpmf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.ppf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.isf(np.nan, 0.5)))