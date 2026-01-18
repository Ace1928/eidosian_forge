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
def make_hs071_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([0, 1, 2, 3], bounds=(1.0, 5.0))
    m.x[0] = 1.0
    m.x[1] = 5.0
    m.x[2] = 5.0
    m.x[3] = 1.0
    m.obj = pyo.Objective(expr=m.x[0] * m.x[3] * (m.x[0] + m.x[1] + m.x[2]) + m.x[2])
    trivial_expr_with_eval_error = pyo.sqrt(1.1 - m.x[0]) ** 2 + m.x[0] - 1.1
    m.ineq1 = pyo.Constraint(expr=m.x[0] * m.x[1] * m.x[2] * m.x[3] >= 25.0)
    m.eq1 = pyo.Constraint(expr=m.x[0] ** 2 + m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 2 == 40.0 + trivial_expr_with_eval_error)
    return m