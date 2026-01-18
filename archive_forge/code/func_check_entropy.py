import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_entropy(distfn, arg, msg):
    ent = distfn.entropy(*arg)
    npt.assert_(not np.isnan(ent), msg + 'test Entropy is nan')