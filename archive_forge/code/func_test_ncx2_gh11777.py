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
def test_ncx2_gh11777():
    df = 6700
    nc = 5300
    x = np.linspace(stats.ncx2.ppf(0.001, df, nc), stats.ncx2.ppf(0.999, df, nc), num=10000)
    ncx2_pdf = stats.ncx2.pdf(x, df, nc)
    gauss_approx = stats.norm.pdf(x, df + nc, np.sqrt(2 * df + 4 * nc))
    assert_allclose(ncx2_pdf, gauss_approx, atol=0.0001)