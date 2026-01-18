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
def test_genextreme_give_no_warnings():
    """regression test for gh-6219"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        stats.genextreme.cdf(0.5, 0)
        stats.genextreme.pdf(0.5, 0)
        stats.genextreme.ppf(0.5, 0)
        stats.genextreme.logpdf(-np.inf, 0.0)
        number_of_warnings_thrown = len(w)
        assert_equal(number_of_warnings_thrown, 0)