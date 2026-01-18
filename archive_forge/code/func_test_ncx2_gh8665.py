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
def test_ncx2_gh8665():
    x = np.array([4.99515382, 10.7617327, 23.1854502, 49.9515382, 107.617327, 231.854502, 499.515382, 1076.17327, 2318.54502, 4995.15382, 10761.7327, 23185.4502, 49951.5382])
    nu, lam = (20, 499.51538166556196)
    sf = stats.ncx2.sf(x, df=nu, nc=lam)
    sf_expected = [1.0, 1.0, 1.0, 1.0, 1.0, 0.9999999999999888, 0.664652558213546, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert_allclose(sf, sf_expected, atol=1e-12)