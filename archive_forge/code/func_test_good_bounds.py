import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem
from scipy._lib._util import VisibleDeprecationWarning
from copy import deepcopy
from datetime import date
def test_good_bounds():
    lp = _LPProblem(c=[1, 2])
    lp_cleaned = _clean_inputs(lp)
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)
    lp_cleaned = _clean_inputs(lp._replace(bounds=[]))
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)
    lp_cleaned = _clean_inputs(lp._replace(bounds=[[]]))
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)
    lp_cleaned = _clean_inputs(lp._replace(bounds=(1, 2)))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 2)
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, 2)]))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 2)
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, None)]))
    assert_equal(lp_cleaned.bounds, [(1, np.inf)] * 2)
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, 1)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, 1)] * 2)
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, None), (-np.inf, None)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, np.inf)] * 2)
    lp = _LPProblem(c=[1, 2, 3, 4])
    lp_cleaned = _clean_inputs(lp)
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 4)
    lp_cleaned = _clean_inputs(lp._replace(bounds=(1, 2)))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 4)
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, 2)]))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 4)
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, None)]))
    assert_equal(lp_cleaned.bounds, [(1, np.inf)] * 4)
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, 1)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, 1)] * 4)
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, None), (-np.inf, None), (None, np.inf), (-np.inf, np.inf)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, np.inf)] * 4)