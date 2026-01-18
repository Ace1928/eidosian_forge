import numpy as np
import pytest
import cvxpy as cvx
import cvxpy.problems.iterative as iterative
import cvxpy.settings as s
from cvxpy.lin_ops.tree_mat import prune_constants
from cvxpy.tests.base_test import BaseTest
def prob_mat_vs_mul_funcs(self, prob) -> None:
    data, dims = prob.get_problem_data(solver=cvx.SCS)
    A = data['A']
    objective, constr_map, dims, solver = prob.canonicalize(cvx.SCS)
    all_ineq = constr_map[s.EQ] + constr_map[s.LEQ]
    var_offsets, var_sizes, x_length = prob._get_var_offsets(objective, all_ineq)
    constraints = constr_map[s.EQ] + constr_map[s.LEQ]
    constraints = prune_constants(constraints)
    Amul, ATmul = iterative.get_mul_funcs(constraints, dims, var_offsets, var_sizes, x_length)
    vec = np.array(range(1, x_length + 1))
    result = np.zeros(A.shape[0])
    Amul(vec, result)
    self.assertItemsAlmostEqual(A @ vec, result)
    Amul(vec, result)
    self.assertItemsAlmostEqual(2 * A @ vec, result)
    vec = np.array(range(A.shape[0]))
    result = np.zeros(A.shape[1])
    ATmul(vec, result)
    self.assertItemsAlmostEqual(A.T @ vec, result)
    ATmul(vec, result)
    self.assertItemsAlmostEqual(2 * A.T @ vec, result)