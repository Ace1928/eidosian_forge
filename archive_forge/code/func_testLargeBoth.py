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
@pytest.mark.slow
def testLargeBoth(self):
    n1, n2 = (10000, 11000)
    lcm = n1 * 11.0
    delta = 1.0 / n1 / n2 / 2 / 2
    x = np.linspace(1, 200, n1) - delta
    y = np.linspace(2, 200, n2)
    self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990660108966576, mode='asymp')
    self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990456491488628, mode='exact')
    self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990660108966576, mode='auto')
    self._testOne(x, y, 'greater', 563.0 / lcm, 0.7561851877420673)
    self._testOne(x, y, 'less', 10.0 / lcm, 0.9998239693191724)
    with suppress_warnings() as sup:
        message = 'ks_2samp: Exact calculation unsuccessful.'
        sup.filter(RuntimeWarning, message)
        self._testOne(x, y, 'greater', 563.0 / lcm, 0.7561851877420673, mode='exact')
        self._testOne(x, y, 'less', 10.0 / lcm, 0.9998239693191724, mode='exact')