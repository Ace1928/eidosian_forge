import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from scipy.optimize import (NonlinearConstraint, LinearConstraint,
from .test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
def test_constraint_dictionary_1(self):

    def fun(x):
        return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2}, {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6}, {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'delta_grad == 0.0')
        res = minimize(fun, self.x0, method=self.method, bounds=self.bnds, constraints=cons)
    assert_allclose(res.x, [1.4, 1.7], rtol=0.0001)
    assert_allclose(res.fun, 0.8, rtol=0.0001)