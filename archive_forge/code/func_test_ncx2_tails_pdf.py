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
def test_ncx2_tails_pdf():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        assert_equal(stats.ncx2.pdf(1, np.arange(340, 350), 2), 0)
        logval = stats.ncx2.logpdf(1, np.arange(340, 350), 2)
    assert_(np.isneginf(logval).all())
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        assert_equal(stats.ncx2.pdf(10000, 3, 12), 0)
        assert_allclose(stats.ncx2.logpdf(10000, 3, 12), -4662.444377524883)