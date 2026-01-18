import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.optimize._constraints import (NonlinearConstraint, Bounds,
from scipy.optimize._trustregion_constr.canonical_constraint \
def test_initial_constraints_as_canonical():
    rng = np.random.RandomState(0)
    x0 = np.array([0.5, 0.4, 0.3, 0.2])
    n = len(x0)
    lb1 = [-1, -np.inf, -2, 3]
    ub1 = [1, np.inf, np.inf, 3]
    bounds = Bounds(lb1, ub1, [False, False, True, False])
    fun, jac, hess = create_quadratic_function(n, 5, rng)
    lb2 = [-10, 3, -np.inf, -np.inf, -5]
    ub2 = [10, 3, np.inf, 5, np.inf]
    nonlinear = NonlinearConstraint(fun, lb2, ub2, jac, hess, [True, False, False, True, False])
    for sparse_jacobian in [False, True]:
        bounds_prepared = PreparedConstraint(bounds, x0, sparse_jacobian)
        nonlinear_prepared = PreparedConstraint(nonlinear, x0, sparse_jacobian)
        f1 = bounds_prepared.fun.f
        J1 = bounds_prepared.fun.J
        f2 = nonlinear_prepared.fun.f
        J2 = nonlinear_prepared.fun.J
        c_eq, c_ineq, J_eq, J_ineq = initial_constraints_as_canonical(n, [bounds_prepared, nonlinear_prepared], sparse_jacobian)
        assert_array_equal(c_eq, [f1[3] - lb1[3], f2[1] - lb2[1]])
        assert_array_equal(c_ineq, [lb1[2] - f1[2], f1[0] - ub1[0], lb1[0] - f1[0], f2[3] - ub2[3], lb2[4] - f2[4], f2[0] - ub2[0], lb2[0] - f2[0]])
        if sparse_jacobian:
            J1 = J1.toarray()
            J2 = J2.toarray()
            J_eq = J_eq.toarray()
            J_ineq = J_ineq.toarray()
        assert_array_equal(J_eq, np.vstack((J1[3], J2[1])))
        assert_array_equal(J_ineq, np.vstack((-J1[2], J1[0], -J1[0], J2[3], -J2[4], J2[0], -J2[0])))