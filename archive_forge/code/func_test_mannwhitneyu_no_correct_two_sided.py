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
def test_mannwhitneyu_no_correct_two_sided(self):
    u1, p1 = stats.mannwhitneyu(self.X, self.Y, False, alternative='two-sided')
    u2, p2 = stats.mannwhitneyu(self.Y, self.X, False, alternative='two-sided')
    assert_equal(p1, p2)
    assert_equal(u1, 498)
    assert_equal(u2, 102)
    assert_approx_equal(p1, 8.81880199916178e-05, significant=self.significant)