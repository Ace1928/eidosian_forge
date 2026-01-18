import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.optimize._constraints import (NonlinearConstraint, Bounds,
from scipy.optimize._trustregion_constr.canonical_constraint \
def test_concatenation():
    rng = np.random.RandomState(0)
    n = 4
    x0 = rng.rand(n)
    f1 = x0
    J1 = np.eye(n)
    lb1 = [-1, -np.inf, -2, 3]
    ub1 = [1, np.inf, np.inf, 3]
    bounds = Bounds(lb1, ub1, [False, False, True, False])
    fun, jac, hess = create_quadratic_function(n, 5, rng)
    f2 = fun(x0)
    J2 = jac(x0)
    lb2 = [-10, 3, -np.inf, -np.inf, -5]
    ub2 = [10, 3, np.inf, 5, np.inf]
    nonlinear = NonlinearConstraint(fun, lb2, ub2, jac, hess, [True, False, False, True, False])
    for sparse_jacobian in [False, True]:
        bounds_prepared = PreparedConstraint(bounds, x0, sparse_jacobian)
        nonlinear_prepared = PreparedConstraint(nonlinear, x0, sparse_jacobian)
        c1 = CanonicalConstraint.from_PreparedConstraint(bounds_prepared)
        c2 = CanonicalConstraint.from_PreparedConstraint(nonlinear_prepared)
        c = CanonicalConstraint.concatenate([c1, c2], sparse_jacobian)
        assert_equal(c.n_eq, 2)
        assert_equal(c.n_ineq, 7)
        c_eq, c_ineq = c.fun(x0)
        assert_array_equal(c_eq, [f1[3] - lb1[3], f2[1] - lb2[1]])
        assert_array_equal(c_ineq, [lb1[2] - f1[2], f1[0] - ub1[0], lb1[0] - f1[0], f2[3] - ub2[3], lb2[4] - f2[4], f2[0] - ub2[0], lb2[0] - f2[0]])
        J_eq, J_ineq = c.jac(x0)
        if sparse_jacobian:
            J_eq = J_eq.toarray()
            J_ineq = J_ineq.toarray()
        assert_array_equal(J_eq, np.vstack((J1[3], J2[1])))
        assert_array_equal(J_ineq, np.vstack((-J1[2], J1[0], -J1[0], J2[3], -J2[4], J2[0], -J2[0])))
        v_eq = rng.rand(c.n_eq)
        v_ineq = rng.rand(c.n_ineq)
        v = np.zeros(5)
        v[1] = v_eq[1]
        v[3] = v_ineq[3]
        v[4] = -v_ineq[4]
        v[0] = v_ineq[5] - v_ineq[6]
        H = c.hess(x0, v_eq, v_ineq).dot(np.eye(n))
        assert_array_equal(H, hess(x0, v))
        assert_array_equal(c.keep_feasible, [True, False, False, True, False, True, True])