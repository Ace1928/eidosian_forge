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
@pytest.mark.parametrize('x, a, expected', [(-20, 1, 1.030576811219279e-09), (-40, 1, 2.1241771276457944e-18), (-50, 5, 2.7248509914602648e-17), (-25, 0.125, 5.333071920958156e-14), (5, 1, 0.9966310265004573)])
def test_cdf_ppf_sf_isf_tail(self, x, a, expected):
    cdf = stats.dgamma.cdf(x, a)
    assert_allclose(cdf, expected, rtol=5e-15)
    ppf = stats.dgamma.ppf(expected, a)
    assert_allclose(ppf, x, rtol=5e-15)
    sf = stats.dgamma.sf(-x, a)
    assert_allclose(sf, expected, rtol=5e-15)
    isf = stats.dgamma.isf(expected, a)
    assert_allclose(isf, -x, rtol=5e-15)