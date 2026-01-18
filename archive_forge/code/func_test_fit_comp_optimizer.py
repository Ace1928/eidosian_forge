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
@pytest.mark.parametrize('dist', [stats.gumbel_r, stats.gumbel_l])
@pytest.mark.parametrize('loc_rvs', [-1, 0, 1])
@pytest.mark.parametrize('scale_rvs', [0.1, 1, 5])
@pytest.mark.parametrize('fix_loc, fix_scale', ([True, False], [False, True]))
def test_fit_comp_optimizer(self, dist, loc_rvs, scale_rvs, fix_loc, fix_scale, rng):
    data = dist.rvs(size=100, loc=loc_rvs, scale=scale_rvs, random_state=rng)
    kwds = dict()
    if fix_loc:
        kwds['floc'] = loc_rvs * 2
    if fix_scale:
        kwds['fscale'] = scale_rvs * 2
    _assert_less_or_close_loglike(dist, data, **kwds)