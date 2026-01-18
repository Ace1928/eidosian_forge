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
@pytest.mark.parametrize('a,b', [(2, 3), (8, 4), (12, 13)])
def test_compare_with_gamlss_r(self, gamlss_pdf_data, a, b):
    """Compare the pdf with a table of reference values. The table of
        reference values was produced using R, where the Jones and Faddy skew
        t distribution is available in the GAMLSS package as `ST5`.
        """
    data = gamlss_pdf_data[(gamlss_pdf_data['a'] == a) & (gamlss_pdf_data['b'] == b)]
    x, pdf = (data['x'], data['pdf'])
    assert_allclose(pdf, stats.jf_skew_t(a, b).pdf(x), rtol=1e-12)