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
def testTwoVsThree(self):
    data1 = np.array([1.0, 2.0])
    data1p = data1 + 0.01
    data1m = data1 - 0.01
    data2 = np.array([1.0, 2.0, 3.0])
    self._testOne(data1p, data2, 'two-sided', 1.0 / 3, 1.0)
    self._testOne(data1p, data2, 'greater', 1.0 / 3, 0.7)
    self._testOne(data1p, data2, 'less', 1.0 / 3, 0.7)
    self._testOne(data1m, data2, 'two-sided', 2.0 / 3, 0.6)
    self._testOne(data1m, data2, 'greater', 2.0 / 3, 0.3)
    self._testOne(data1m, data2, 'less', 0, 1.0)