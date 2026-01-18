import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
from pyomo.environ import (
from pyomo.core.expr import MonomialTermExpression
from pyomo.core.expr.ndarray import NumericNDArray
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr.compare import compare_expressions
from pyomo.core.expr.relational_expr import InequalityExpression
from pyomo.repn import generate_standard_repn
def test_create_objective_from_numpy(self):
    model = ConcreteModel()
    nsample = 3
    nvariables = 2
    X0 = np.array(range(nsample)).reshape([nsample, 1])
    model.X = 1 + np.array(range(nsample * nvariables)).reshape((nsample, nvariables))
    X = np.concatenate([X0, model.X], axis=1)
    model.I = RangeSet(1, nsample)
    model.J = RangeSet(1, nvariables)
    error = np.ones((nsample, 1))
    beta = np.ones((nvariables + 1, 1))
    model.Y = np.dot(X, beta) + error
    model.beta = Var(model.J)
    model.beta0 = Var()

    def obj_fun(model):
        return sum((abs(model.Y[i - 1] - (model.beta0 + sum((model.X[i - 1, j - 1] * model.beta[j] for j in model.J)))) for i in model.I))
    model.OBJ = Objective(rule=obj_fun)

    def obj_fun_quad(model):
        return sum(((model.Y[i - 1] - (model.beta0 + sum((model.X[i - 1, j - 1] * model.beta[j] for j in model.J)))) ** 2 for i in model.I))
    model.OBJ_QUAD = Objective(rule=obj_fun_quad)
    self.assertEqual(str(model.OBJ.expr), 'abs(4.0 - (beta[1] + 2*beta[2] + beta0)) + abs(9.0 - (3*beta[1] + 4*beta[2] + beta0)) + abs(14.0 - (5*beta[1] + 6*beta[2] + beta0))')
    self.assertEqual(model.OBJ.expr.polynomial_degree(), None)
    self.assertEqual(model.OBJ_QUAD.expr.polynomial_degree(), 2)