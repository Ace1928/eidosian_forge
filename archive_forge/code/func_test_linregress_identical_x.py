import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
def test_linregress_identical_x():
    x = np.zeros(10)
    y = np.random.random(10)
    msg = 'Cannot calculate a linear regression if all x values are identical'
    with assert_raises(ValueError, match=msg):
        mstats.linregress(x, y)