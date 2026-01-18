import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
def test_21_3_hess_options_rosen(self):
    """Ensure the Hessian gets passed correctly to the local minimizer
        routine. Previous report gh-14533.
        """
    bounds = [(0, 1.6), (0, 1.6), (0, 1.4), (0, 1.4), (0, 1.4)]
    options = {'jac': rosen_der, 'hess': rosen_hess}
    minimizer_kwargs = {'method': 'Newton-CG'}
    res = shgo(rosen, bounds, minimizer_kwargs=minimizer_kwargs, options=options)
    ref = minimize(rosen, numpy.zeros(5), method='Newton-CG', **options)
    assert res.success
    assert_allclose(res.fun, ref.fun)
    assert_allclose(res.x, ref.x, atol=1e-15)