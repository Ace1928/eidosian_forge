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
def test_compare_SAS(self):
    a = [12, 14, 18, 25, 32, 44, 12, 14, 18, 25, 32, 44]
    b = [17, 22, 14, 12, 30, 29, 19, 17, 22, 14, 12, 30, 29, 19]
    statistic, pvalue = stats.ttest_ind(a, b, trim=0.09, equal_var=False)
    assert_allclose(pvalue, 0.514522, atol=1e-06)
    assert_allclose(statistic, 0.669169, atol=1e-06)