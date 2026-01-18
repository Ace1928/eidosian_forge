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
def test_equal_var(self):
    """
        The PairedData library only supports unequal variances. To compare
        samples with equal variances, the multicon library is used.
        > library(multicon)
        > a <- c(2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9)
        > b <- c(6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1)
        > dv = c(a,b)
        > iv = c(rep('a', length(a)), rep('b', length(b)))
        > yuenContrast(dv~ iv, EQVAR = TRUE)
        $Ms
           N                 M wgt
        a 11 2.442857142857143   1
        b 11 5.385714285714286  -1

        $test
                              stat df              crit                   p
        results -4.246116897032513 12 2.178812829667228 0.00113508833897713
        """
    a = [2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9]
    b = [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1]
    statistic, pvalue = stats.ttest_ind(a, b, trim=0.2)
    assert_allclose(pvalue, 0.00113508833897713, atol=1e-10)
    assert_allclose(statistic, -4.246116897032513, atol=1e-10)