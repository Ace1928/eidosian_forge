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
def test_logpdf_pdf(self):
    x = [1, 1000.0, 10, 1]
    df = [1e+100, 1e+50, 1e+20, 1]
    logpdf_ref = [-1.4189385332046727, -500000.9189385332, -50.918938533204674, -1.8378770664093456]
    pdf_ref = [0.24197072451914334, 0, 7.69459862670642e-23, 0.15915494309189535]
    assert_allclose(stats.t.logpdf(x, df), logpdf_ref, rtol=1e-15)
    assert_allclose(stats.t.pdf(x, df), pdf_ref, rtol=1e-14)