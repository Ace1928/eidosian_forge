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
def test_compare_dtypes(self):
    args = [[13, 13, 13, 13, 13, 13, 13, 12, 12], [14, 13, 12, 12, 12, 12, 12, 11, 11], [14, 14, 13, 13, 13, 13, 13, 12, 12], [15, 14, 13, 13, 13, 12, 12, 12, 11]]
    args_int16 = np.array(args, dtype=np.int16)
    args_int32 = np.array(args, dtype=np.int32)
    args_uint8 = np.array(args, dtype=np.uint8)
    args_float64 = np.array(args, dtype=np.float64)
    res_int16 = stats.alexandergovern(*args_int16)
    res_int32 = stats.alexandergovern(*args_int32)
    res_unit8 = stats.alexandergovern(*args_uint8)
    res_float64 = stats.alexandergovern(*args_float64)
    assert res_int16.pvalue == res_int32.pvalue == res_unit8.pvalue == res_float64.pvalue
    assert res_int16.statistic == res_int32.statistic == res_unit8.statistic == res_float64.statistic