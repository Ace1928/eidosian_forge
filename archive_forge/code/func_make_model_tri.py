import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.interior_point.interior_point import InteriorPointSolver
from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface
def make_model_tri(n, small_val=1e-07, big_val=100.0):
    m = ConcreteModel()
    m.x = Var(range(n), initialize=0.5)

    def c_rule(m, i):
        return big_val * m.x[i - 1] + small_val * m.x[i] + big_val * m.x[i + 1] == 1
    m.c = Constraint(range(1, n - 1), rule=c_rule)
    m.obj = Objective(expr=small_val * sum(((m.x[i] - 1) ** 2 for i in range(n))))
    return m