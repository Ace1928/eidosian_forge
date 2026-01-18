import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_pickling(distfn, args):
    rndm = distfn.random_state
    distfn.random_state = 1234
    distfn.rvs(*args, size=8)
    s = pickle.dumps(distfn)
    r0 = distfn.rvs(*args, size=8)
    unpickled = pickle.loads(s)
    r1 = unpickled.rvs(*args, size=8)
    npt.assert_equal(r0, r1)
    medians = [distfn.ppf(0.5, *args), unpickled.ppf(0.5, *args)]
    npt.assert_equal(medians[0], medians[1])
    npt.assert_equal(distfn.cdf(medians[0], *args), unpickled.cdf(medians[1], *args))
    frozen_dist = distfn(*args)
    pkl = pickle.dumps(frozen_dist)
    unpickled = pickle.loads(pkl)
    r0 = frozen_dist.rvs(size=8)
    r1 = unpickled.rvs(size=8)
    npt.assert_equal(r0, r1)
    if hasattr(distfn, 'fit'):
        fit_function = distfn.fit
        pickled_fit_function = pickle.dumps(fit_function)
        unpickled_fit_function = pickle.loads(pickled_fit_function)
        assert fit_function.__name__ == unpickled_fit_function.__name__ == 'fit'
    distfn.random_state = rndm