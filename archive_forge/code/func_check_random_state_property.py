import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_random_state_property(distfn, args):
    rndm = distfn.random_state
    np.random.seed(1234)
    distfn.random_state = None
    r0 = distfn.rvs(*args, size=8)
    distfn.random_state = 1234
    r1 = distfn.rvs(*args, size=8)
    npt.assert_equal(r0, r1)
    distfn.random_state = np.random.RandomState(1234)
    r2 = distfn.rvs(*args, size=8)
    npt.assert_equal(r0, r2)
    if hasattr(np.random, 'default_rng'):
        rng = np.random.default_rng(1234)
        distfn.rvs(*args, size=1, random_state=rng)
    distfn.random_state = 2
    orig_state = distfn.random_state.get_state()
    r3 = distfn.rvs(*args, size=8, random_state=np.random.RandomState(1234))
    npt.assert_equal(r0, r3)
    npt.assert_equal(distfn.random_state.get_state(), orig_state)
    distfn.random_state = rndm