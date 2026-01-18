import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem
from scipy._lib._util import VisibleDeprecationWarning
from copy import deepcopy
from datetime import date
def test__clean_inputs1():
    lp = _LPProblem(c=[1, 2], A_ub=[[1, 1], [2, 2]], b_ub=[1, 1], A_eq=[[1, 1], [2, 2]], b_eq=[1, 1], bounds=None)
    lp_cleaned = _clean_inputs(lp)
    assert_allclose(lp_cleaned.c, np.array(lp.c))
    assert_allclose(lp_cleaned.A_ub, np.array(lp.A_ub))
    assert_allclose(lp_cleaned.b_ub, np.array(lp.b_ub))
    assert_allclose(lp_cleaned.A_eq, np.array(lp.A_eq))
    assert_allclose(lp_cleaned.b_eq, np.array(lp.b_eq))
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)
    assert_(lp_cleaned.c.shape == (2,), '')
    assert_(lp_cleaned.A_ub.shape == (2, 2), '')
    assert_(lp_cleaned.b_ub.shape == (2,), '')
    assert_(lp_cleaned.A_eq.shape == (2, 2), '')
    assert_(lp_cleaned.b_eq.shape == (2,), '')