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
def test_cdf_ppf(self):
    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
    cdf_values = np.asarray([0.0 / 25.0, 0.0 / 25.0, 0.0 / 25.0, 0.5 / 25.0, 1.0 / 25.0, 2.0 / 25.0, 3.0 / 25.0, 4.5 / 25.0, 6.0 / 25.0, 8.0 / 25.0, 10.0 / 25.0, 12.5 / 25.0, 15.0 / 25.0, 17.0 / 25.0, 19.0 / 25.0, 20.5 / 25.0, 22.0 / 25.0, 23.5 / 25.0, 25.0 / 25.0, 25.0 / 25.0])
    assert_allclose(self.template.cdf(values), cdf_values)
    assert_allclose(self.template.ppf(cdf_values[2:-1]), values[2:-1])
    x = np.linspace(1.0, 9.0, 100)
    assert_allclose(self.template.ppf(self.template.cdf(x)), x)
    x = np.linspace(0.0, 1.0, 100)
    assert_allclose(self.template.cdf(self.template.ppf(x)), x)
    x = np.linspace(-2, 2, 10)
    assert_allclose(self.norm_template.cdf(x), stats.norm.cdf(x, loc=1.0, scale=2.5), rtol=0.1)