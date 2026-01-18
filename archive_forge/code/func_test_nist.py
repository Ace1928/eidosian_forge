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
def test_nist(self):
    filenames = ['SiRstv.dat', 'SmLs01.dat', 'SmLs02.dat', 'SmLs03.dat', 'AtmWtAg.dat', 'SmLs04.dat', 'SmLs05.dat', 'SmLs06.dat', 'SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat']
    for test_case in filenames:
        rtol = 1e-07
        fname = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/nist_anova', test_case))
        with open(fname) as f:
            content = f.read().split('\n')
        certified = [line.split() for line in content[40:48] if line.strip()]
        dataf = np.loadtxt(fname, skiprows=60)
        y, x = dataf.T
        y = y.astype(int)
        caty = np.unique(y)
        f = float(certified[0][-1])
        xlist = [x[y == i] for i in caty]
        res = stats.f_oneway(*xlist)
        hard_tc = ('SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat')
        if test_case in hard_tc:
            rtol = 0.0001
        assert_allclose(res[0], f, rtol=rtol, err_msg='Failing testcase: %s' % test_case)