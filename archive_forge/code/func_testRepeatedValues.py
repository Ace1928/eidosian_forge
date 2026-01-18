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
def testRepeatedValues(self):
    x2233 = np.array([2] * 3 + [3] * 4 + [5] * 5 + [6] * 4, dtype=int)
    x3344 = x2233 + 1
    x2356 = np.array([2] * 3 + [3] * 4 + [5] * 10 + [6] * 4, dtype=int)
    x3467 = np.array([3] * 10 + [4] * 2 + [6] * 10 + [7] * 4, dtype=int)
    self._testOne(x2233, x3344, 'two-sided', 5.0 / 16, 0.4262934613454952)
    self._testOne(x2233, x3344, 'greater', 5.0 / 16, 0.21465428276573786)
    self._testOne(x2233, x3344, 'less', 0.0 / 16, 1.0)
    self._testOne(x2356, x3467, 'two-sided', 190.0 / 21 / 26, 0.0919245790168125)
    self._testOne(x2356, x3467, 'greater', 190.0 / 21 / 26, 0.0459633806858544)
    self._testOne(x2356, x3467, 'less', 70.0 / 21 / 26, 0.6121593130022775)