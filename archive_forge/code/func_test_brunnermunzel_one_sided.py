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
def test_brunnermunzel_one_sided(self):
    u1, p1 = stats.brunnermunzel(self.X, self.Y, alternative='less')
    u2, p2 = stats.brunnermunzel(self.Y, self.X, alternative='greater')
    u3, p3 = stats.brunnermunzel(self.X, self.Y, alternative='greater')
    u4, p4 = stats.brunnermunzel(self.Y, self.X, alternative='less')
    assert_approx_equal(p1, p2, significant=self.significant)
    assert_approx_equal(p3, p4, significant=self.significant)
    assert_(p1 != p3)
    assert_approx_equal(u1, 3.1374674823029505, significant=self.significant)
    assert_approx_equal(u2, -3.1374674823029505, significant=self.significant)
    assert_approx_equal(u3, 3.1374674823029505, significant=self.significant)
    assert_approx_equal(u4, -3.1374674823029505, significant=self.significant)
    assert_approx_equal(p1, 0.002893104333075734, significant=self.significant)
    assert_approx_equal(p3, 0.9971068956669242, significant=self.significant)