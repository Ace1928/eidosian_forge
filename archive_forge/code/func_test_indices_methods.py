import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP, AmplNLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import tempfile
from pyomo.contrib.pynumero.interfaces.utils import (
def test_indices_methods(self):
    nlp = PyomoNLP(self.pm)
    variables = nlp.get_pyomo_variables()
    expected_ids = [id(self.pm.x[i]) for i in range(1, 10)]
    ids = [id(variables[i]) for i in range(9)]
    self.assertTrue(expected_ids == ids)
    variable_names = nlp.variable_names()
    expected_names = [self.pm.x[i].getname() for i in range(1, 10)]
    self.assertTrue(variable_names == expected_names)
    constraints = nlp.get_pyomo_constraints()
    expected_ids = [id(self.pm.c[i]) for i in range(1, 10)]
    ids = [id(constraints[i]) for i in range(9)]
    self.assertTrue(expected_ids == ids)
    constraint_names = nlp.constraint_names()
    expected_names = [c.getname() for c in nlp.get_pyomo_constraints()]
    self.assertTrue(constraint_names == expected_names)
    eq_constraints = nlp.get_pyomo_equality_constraints()
    eq_indices = [2, 6]
    expected_eq_ids = [id(self.pm.c[i]) for i in eq_indices]
    eq_ids = [id(con) for con in eq_constraints]
    self.assertEqual(eq_ids, expected_eq_ids)
    eq_constraint_names = nlp.equality_constraint_names()
    expected_eq_names = [c.getname(fully_qualified=True) for c in nlp.get_pyomo_equality_constraints()]
    self.assertEqual(eq_constraint_names, expected_eq_names)
    ineq_constraints = nlp.get_pyomo_inequality_constraints()
    ineq_indices = [1, 3, 4, 5, 7, 8, 9]
    expected_ineq_ids = [id(self.pm.c[i]) for i in ineq_indices]
    ineq_ids = [id(con) for con in ineq_constraints]
    self.assertEqual(eq_ids, expected_eq_ids)
    expected_primal_indices = [i for i in range(9)]
    self.assertTrue(expected_primal_indices == nlp.get_primal_indices([self.pm.x]))
    expected_primal_indices = [0, 3, 8, 4]
    variables = [self.pm.x[1], self.pm.x[4], self.pm.x[9], self.pm.x[5]]
    self.assertTrue(expected_primal_indices == nlp.get_primal_indices(variables))
    expected_constraint_indices = [i for i in range(9)]
    self.assertTrue(expected_constraint_indices == nlp.get_constraint_indices([self.pm.c]))
    expected_constraint_indices = [0, 3, 8, 4]
    constraints = [self.pm.c[1], self.pm.c[4], self.pm.c[9], self.pm.c[5]]
    self.assertTrue(expected_constraint_indices == nlp.get_constraint_indices(constraints))
    pyomo_eq_indices = [2, 6]
    with self.assertRaises(KeyError):
        nlp.get_equality_constraint_indices([self.pm.c])
    eq_constraints = [self.pm.c[i] for i in pyomo_eq_indices]
    expected_eq_indices = [0, 1]
    eq_constraint_indices = nlp.get_equality_constraint_indices(eq_constraints)
    self.assertEqual(expected_eq_indices, eq_constraint_indices)
    pyomo_ineq_indices = [1, 3, 4, 5, 7, 9]
    with self.assertRaises(KeyError):
        nlp.get_inequality_constraint_indices([self.pm.c])
    ineq_constraints = [self.pm.c[i] for i in pyomo_ineq_indices]
    expected_ineq_indices = [0, 1, 2, 3, 4, 6]
    ineq_constraint_indices = nlp.get_inequality_constraint_indices(ineq_constraints)
    self.assertEqual(expected_ineq_indices, ineq_constraint_indices)
    expected_gradient = np.asarray([2 * sum(((i + 1) * (j + 1) for j in range(9))) for i in range(9)], dtype=np.float64)
    grad_obj = nlp.extract_subvector_grad_objective([self.pm.x])
    self.assertTrue(np.array_equal(expected_gradient, grad_obj))
    expected_gradient = np.asarray([2 * sum(((i + 1) * (j + 1) for j in range(9))) for i in [0, 3, 8, 4]], dtype=np.float64)
    variables = [self.pm.x[1], self.pm.x[4], self.pm.x[9], self.pm.x[5]]
    grad_obj = nlp.extract_subvector_grad_objective(variables)
    self.assertTrue(np.array_equal(expected_gradient, grad_obj))
    expected_con = np.asarray([45, 88, 3 * 45, 4 * 45, 5 * 45, 276, 7 * 45, 8 * 45, 9 * 45], dtype=np.float64)
    con = nlp.extract_subvector_constraints([self.pm.c])
    self.assertTrue(np.array_equal(expected_con, con))
    expected_con = np.asarray([45, 4 * 45, 9 * 45, 5 * 45], dtype=np.float64)
    constraints = [self.pm.c[1], self.pm.c[4], self.pm.c[9], self.pm.c[5]]
    con = nlp.extract_subvector_constraints(constraints)
    self.assertTrue(np.array_equal(expected_con, con))
    expected_jac = [[i * j for j in range(1, 10)] for i in range(1, 10)]
    expected_jac = np.asarray(expected_jac, dtype=np.float64)
    jac = nlp.extract_submatrix_jacobian(pyomo_variables=[self.pm.x], pyomo_constraints=[self.pm.c])
    dense_jac = jac.todense()
    self.assertTrue(np.array_equal(dense_jac, expected_jac))
    expected_jac = [[i * j for j in [1, 4, 9, 5]] for i in [2, 6, 4]]
    expected_jac = np.asarray(expected_jac, dtype=np.float64)
    variables = [self.pm.x[1], self.pm.x[4], self.pm.x[9], self.pm.x[5]]
    constraints = [self.pm.c[2], self.pm.c[6], self.pm.c[4]]
    jac = nlp.extract_submatrix_jacobian(pyomo_variables=variables, pyomo_constraints=constraints)
    dense_jac = jac.todense()
    self.assertTrue(np.array_equal(dense_jac, expected_jac))
    expected_hess = [[2.0 * i * j for j in range(1, 10)] for i in range(1, 10)]
    expected_hess = np.asarray(expected_hess, dtype=np.float64)
    hess = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=[self.pm.x], pyomo_variables_cols=[self.pm.x])
    dense_hess = hess.todense()
    self.assertTrue(np.array_equal(dense_hess, expected_hess))
    expected_hess = [[2.0 * i * j for j in [1, 4, 9, 5]] for i in [1, 4, 9, 5]]
    expected_hess = np.asarray(expected_hess, dtype=np.float64)
    variables = [self.pm.x[1], self.pm.x[4], self.pm.x[9], self.pm.x[5]]
    hess = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=variables, pyomo_variables_cols=variables)
    dense_hess = hess.todense()
    self.assertTrue(np.array_equal(dense_hess, expected_hess))