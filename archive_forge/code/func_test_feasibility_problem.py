import osqp
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
from osqp.tests.utils import load_high_accuracy, rel_tol, abs_tol, decimal_tol
def test_feasibility_problem(self):
    res = self.model.solve()
    x_sol, y_sol, obj_sol = load_high_accuracy('test_feasibility_problem')
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=decimal_tol)