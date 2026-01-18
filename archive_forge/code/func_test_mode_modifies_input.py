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
def test_mode_modifies_input(self):
    im = np.zeros((100, 100))
    im[:50, :] += 1
    im[:, :50] += 1
    cp = im.copy()
    mstats.mode(im, None)
    assert_equal(im, cp)