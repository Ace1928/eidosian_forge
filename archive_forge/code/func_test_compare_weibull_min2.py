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
def test_compare_weibull_min2(self):
    c, a, b = (2.5, 0.25, 1.25)
    x = np.linspace(a, b, 100)
    pdf1 = stats.truncweibull_min.pdf(x, c, a, b)
    cdf1 = stats.truncweibull_min.cdf(x, c, a, b)
    norm = stats.weibull_min.cdf(b, c) - stats.weibull_min.cdf(a, c)
    pdf2 = stats.weibull_min.pdf(x, c) / norm
    cdf2 = (stats.weibull_min.cdf(x, c) - stats.weibull_min.cdf(a, c)) / norm
    np.testing.assert_allclose(pdf1, pdf2)
    np.testing.assert_allclose(cdf1, cdf2)