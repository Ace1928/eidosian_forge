import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
def create_model3(G, A, b, c):
    nx = G.shape[0]
    nl = A.shape[0]
    model = pyo.ConcreteModel()
    model.var_ids = range(nx)
    model.con_ids = range(nl)
    model.x = pyo.Var(model.var_ids, initialize=0.0)
    model.hessian_f = pyo.Param(model.var_ids, model.var_ids, mutable=True, rule=lambda m, i, j: G[i, j])
    model.jacobian_c = pyo.Param(model.con_ids, model.var_ids, mutable=True, rule=lambda m, i, j: A[i, j])
    model.rhs = pyo.Param(model.con_ids, mutable=True, rule=lambda m, i: b[i])
    model.grad_f = pyo.Param(model.var_ids, mutable=True, rule=lambda m, i: c[i])

    def equality_constraint_rule(m, i):
        return sum((m.jacobian_c[i, j] * m.x[j] for j in m.var_ids)) == m.rhs[i]
    model.equalities = pyo.Constraint(model.con_ids, rule=equality_constraint_rule)

    def objective_rule(m):
        accum = 0.0
        for i in m.var_ids:
            accum += m.x[i] * sum((m.hessian_f[i, j] * m.x[j] for j in m.var_ids))
        accum *= 0.5
        accum += sum((m.x[j] * m.grad_f[j] for j in m.var_ids))
        return accum
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    return model