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
def test_fit_gh19332(self):
    x = np.array([-5, -1, 1 / 100000] + 12 * [1] + [5])
    params = stats.skewnorm.fit(x)
    res = stats.skewnorm.nnlf(params, x)
    params_super = stats.skewnorm.fit(x, superfit=True)
    ref = stats.skewnorm.nnlf(params_super, x)
    assert res < ref - 0.5
    rng = np.random.default_rng(9842356982345693637)
    bounds = {'a': (-5, 5), 'loc': (-10, 10), 'scale': (1e-16, 10)}

    def optimizer(fun, bounds):
        return differential_evolution(fun, bounds, seed=rng)
    fit_result = stats.fit(stats.skewnorm, x, bounds, optimizer=optimizer)
    np.testing.assert_allclose(params, fit_result.params, rtol=0.0001)