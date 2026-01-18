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
def test_compare_scholar(self):
    """
        Data taken from 'The Modification and Evaluation of the
        Alexander-Govern Test in Terms of Power' by Kingsley Ochuko, T.,
        Abdullah, S., Binti Zain, Z., & Soaad Syed Yahaya, S. (2015).
        """
    young = [482.43, 484.36, 488.84, 495.15, 495.24, 502.69, 504.62, 518.29, 519.1, 524.1, 524.12, 531.18, 548.42, 572.1, 584.68, 609.09, 609.53, 666.63, 676.4]
    middle = [335.59, 338.43, 353.54, 404.27, 437.5, 469.01, 485.85, 487.3, 493.08, 494.31, 499.1, 886.41]
    old = [519.01, 528.5, 530.23, 536.03, 538.56, 538.83, 557.24, 558.61, 558.95, 565.43, 586.39, 594.69, 629.22, 645.69, 691.84]
    soln = stats.alexandergovern(young, middle, old)
    assert_allclose(soln.statistic, 5.3237, atol=0.001)
    assert_allclose(soln.pvalue, 0.06982, atol=0.0001)
    '\n        > library("onewaytests")\n        > library("tibble")\n        > young <- c(482.43, 484.36, 488.84, 495.15, 495.24, 502.69, 504.62,\n        +                  518.29, 519.1, 524.1, 524.12, 531.18, 548.42, 572.1,\n        +                  584.68, 609.09, 609.53, 666.63, 676.4)\n        > middle <- c(335.59, 338.43, 353.54, 404.27, 437.5, 469.01, 485.85,\n        +                   487.3, 493.08, 494.31, 499.1, 886.41)\n        > old <- c(519.01, 528.5, 530.23, 536.03, 538.56, 538.83, 557.24,\n        +                   558.61, 558.95, 565.43, 586.39, 594.69, 629.22,\n        +                   645.69, 691.84)\n        > young_fct <- c(rep("young", times=19))\n        > middle_fct <-c(rep("middle", times=12))\n        > old_fct <- c(rep("old", times=15))\n        > ag.test(a ~ b, tibble(a=c(young, middle, old), b=factor(c(young_fct,\n        +                                              middle_fct, old_fct))))\n\n        Alexander-Govern Test (alpha = 0.05)\n        -------------------------------------------------------------\n        data : a and b\n\n        statistic  : 5.324629\n        parameter  : 2\n        p.value    : 0.06978651\n\n        Result     : Difference is not statistically significant.\n        -------------------------------------------------------------\n\n        '
    assert_allclose(soln.statistic, 5.324629)
    assert_allclose(soln.pvalue, 0.06978651)