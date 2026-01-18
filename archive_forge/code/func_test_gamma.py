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
def test_gamma(self):
    a = 2.0
    dist = stats.gamma
    frozen = stats.gamma(a)
    result_f = frozen.pdf(20.0)
    result = dist.pdf(20.0, a)
    assert_equal(result_f, result)
    result_f = frozen.cdf(20.0)
    result = dist.cdf(20.0, a)
    assert_equal(result_f, result)
    result_f = frozen.ppf(0.25)
    result = dist.ppf(0.25, a)
    assert_equal(result_f, result)
    result_f = frozen.isf(0.25)
    result = dist.isf(0.25, a)
    assert_equal(result_f, result)
    result_f = frozen.sf(10.0)
    result = dist.sf(10.0, a)
    assert_equal(result_f, result)
    result_f = frozen.median()
    result = dist.median(a)
    assert_equal(result_f, result)
    result_f = frozen.mean()
    result = dist.mean(a)
    assert_equal(result_f, result)
    result_f = frozen.var()
    result = dist.var(a)
    assert_equal(result_f, result)
    result_f = frozen.std()
    result = dist.std(a)
    assert_equal(result_f, result)
    result_f = frozen.entropy()
    result = dist.entropy(a)
    assert_equal(result_f, result)
    result_f = frozen.moment(2)
    result = dist.moment(2, a)
    assert_equal(result_f, result)
    assert_equal(frozen.a, frozen.dist.a)
    assert_equal(frozen.b, frozen.dist.b)