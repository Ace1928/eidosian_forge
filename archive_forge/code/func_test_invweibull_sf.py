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
@pytest.mark.parametrize('x, c, expected', [(3, 1.5, 0.1750645100707133), (2000, 1.5, 1.1180277387731872e-05), (2000, 9.25, 2.9206030883226965e-31), (1000000000000000.0, 1.5, 3.1622776601683793e-23)])
def test_invweibull_sf(x, c, expected):
    computed = stats.invweibull.sf(x, c)
    assert_allclose(computed, expected, rtol=1e-15)