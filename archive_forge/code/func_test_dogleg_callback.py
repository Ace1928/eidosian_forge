import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
from scipy.optimize import (minimize, rosen, rosen_der, rosen_hess,
def test_dogleg_callback(self):
    accumulator = Accumulator()
    maxiter = 5
    r = minimize(rosen, self.hard_guess, jac=rosen_der, hess=rosen_hess, callback=accumulator, method='dogleg', options={'return_all': True, 'maxiter': maxiter})
    assert_equal(accumulator.count, maxiter)
    assert_equal(len(r['allvecs']), maxiter + 1)
    assert_allclose(r['x'], r['allvecs'][-1])
    assert_allclose(sum(r['allvecs'][1:]), accumulator.accum)