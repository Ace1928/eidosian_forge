import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
@pytest.mark.slow
def test_list_of_problems(self):
    list_of_problems = [Maratos(), Maratos(constr_hess='2-point'), Maratos(constr_hess=SR1()), Maratos(constr_jac='2-point', constr_hess=SR1()), MaratosGradInFunc(), HyperbolicIneq(), HyperbolicIneq(constr_hess='3-point'), HyperbolicIneq(constr_hess=BFGS()), HyperbolicIneq(constr_jac='3-point', constr_hess=BFGS()), Rosenbrock(), IneqRosenbrock(), EqIneqRosenbrock(), BoundedRosenbrock(), Elec(n_electrons=2), Elec(n_electrons=2, constr_hess='2-point'), Elec(n_electrons=2, constr_hess=SR1()), Elec(n_electrons=2, constr_jac='3-point', constr_hess=SR1())]
    for prob in list_of_problems:
        for grad in (prob.grad, '3-point', False):
            for hess in (prob.hess, '3-point', SR1(), BFGS(exception_strategy='damp_update'), BFGS(exception_strategy='skip_update')):
                if grad in ('2-point', '3-point', 'cs', False) and hess in ('2-point', '3-point', 'cs'):
                    continue
                if prob.grad is True and grad in ('3-point', False):
                    continue
                with suppress_warnings() as sup:
                    sup.filter(UserWarning, 'delta_grad == 0.0')
                    result = minimize(prob.fun, prob.x0, method='trust-constr', jac=grad, hess=hess, bounds=prob.bounds, constraints=prob.constr)
                if prob.x_opt is not None:
                    assert_array_almost_equal(result.x, prob.x_opt, decimal=5)
                    if result.status == 1:
                        assert_array_less(result.optimality, 1e-08)
                if result.status == 2:
                    assert_array_less(result.tr_radius, 1e-08)
                    if result.method == 'tr_interior_point':
                        assert_array_less(result.barrier_parameter, 1e-08)
                if result.status in (0, 3):
                    raise RuntimeError('Invalid termination condition.')