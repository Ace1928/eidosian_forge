import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from scipy.optimize import (NonlinearConstraint, LinearConstraint,
from .test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
def test_constraint_dictionary_3(self):

    def fun(x):
        return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
    cons = [{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2}, NonlinearConstraint(lambda x: x[0] - x[1], 0, 0)]
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'delta_grad == 0.0')
        res = minimize(fun, self.x0, method=self.method, bounds=self.bnds, constraints=cons)
    assert_allclose(res.x, [1.75, 1.75], rtol=0.0001)
    assert_allclose(res.fun, 1.125, rtol=0.0001)