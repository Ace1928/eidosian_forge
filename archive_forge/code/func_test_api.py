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
def test_api(self):
    d = np.ones((5, 5))
    stats.iqr(d)
    stats.iqr(d, None)
    stats.iqr(d, 1)
    stats.iqr(d, (0, 1))
    stats.iqr(d, None, (10, 90))
    stats.iqr(d, None, (30, 20), 1.0)
    stats.iqr(d, None, (25, 75), 1.5, 'propagate')
    stats.iqr(d, None, (50, 50), 'normal', 'raise', 'linear')
    stats.iqr(d, None, (25, 75), -0.4, 'omit', 'lower', True)