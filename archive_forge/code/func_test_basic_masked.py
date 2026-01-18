import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
def test_basic_masked(self):
    for case in power_div_1d_cases:
        mobs = np.ma.array(case.f_obs)
        self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, None, case.chi2)
        self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 'pearson', case.chi2)
        self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 1, case.chi2)
        self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 'log-likelihood', case.log)
        self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 'mod-log-likelihood', case.mod_log)
        self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 'cressie-read', case.cr)
        self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 2 / 3, case.cr)