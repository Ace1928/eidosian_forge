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
def test100_100(self):
    x100 = np.linspace(1, 100, 100)
    x100_2_p1 = x100 + 2 + 0.1
    x100_2_m1 = x100 + 2 - 0.1
    self._testOne(x100, x100_2_p1, 'two-sided', 3.0 / 100, 0.9999999999962055)
    self._testOne(x100, x100_2_p1, 'greater', 3.0 / 100, 0.9143290114276248)
    self._testOne(x100, x100_2_p1, 'less', 0, 1.0)
    self._testOne(x100, x100_2_m1, 'two-sided', 2.0 / 100, 1.0)
    self._testOne(x100, x100_2_m1, 'greater', 2.0 / 100, 0.960978450786184)
    self._testOne(x100, x100_2_m1, 'less', 0, 1.0)