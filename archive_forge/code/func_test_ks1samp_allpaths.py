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
def test_ks1samp_allpaths(self):
    assert_(np.isnan(kolmogn(np.nan, 1, True)))
    with assert_raises(ValueError, match='n is not integral: 1.5'):
        kolmogn(1.5, 1, True)
    assert_(np.isnan(kolmogn(-1, 1, True)))
    dataset = np.asarray([(101, 1, True, 1.0), (101, 1.1, True, 1.0), (101, 0, True, 0.0), (101, -0.1, True, 0.0), (32, 1.0 / 64, True, 0.0), (32, 1.0 / 64, False, 1.0), (32, 0.5, True, 0.9999999363163307), (32, 0.5, False, 6.368366937916623e-08), (32, 1.0 / 8, True, 0.34624229979775223), (32, 1.0 / 4, True, 0.9699508336558085), (1600, 0.49, False, 0.0), (1600, 1 / 16.0, False, 7.0837876229702195e-06), (1600, 14 / 1600, False, 0.99962357317602), (1600, 1 / 32, False, 0.08603386296651416)])
    FuncData(kolmogn, dataset, (0, 1, 2), 3).check(dtypes=[int, float, bool])