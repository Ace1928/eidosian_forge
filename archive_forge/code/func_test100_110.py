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
def test100_110(self):
    x100 = np.linspace(1, 100, 100)
    x110 = np.linspace(1, 100, 110)
    x110_20_p1 = x110 + 20 + 0.1
    x110_20_m1 = x110 + 20 - 0.1
    self._testOne(x100, x110_20_p1, 'two-sided', 232.0 / 1100, 0.015739183865607353)
    self._testOne(x100, x110_20_p1, 'greater', 232.0 / 1100, 0.007869594319053203)
    self._testOne(x100, x110_20_p1, 'less', 0, 1)
    self._testOne(x100, x110_20_m1, 'two-sided', 229.0 / 1100, 0.017803803861026313)
    self._testOne(x100, x110_20_m1, 'greater', 229.0 / 1100, 0.008901905958245056)
    self._testOne(x100, x110_20_m1, 'less', 0.0, 1.0)