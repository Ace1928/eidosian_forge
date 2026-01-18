import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
def get_solution(self):
    try:
        import numpy as np
    except:
        raise unittest.SkipTest('numpy is not available')
    p1 = self.m.p1.value
    p2 = self.m.p2.value
    p3 = self.m.p3.value
    p4 = self.m.p4.value
    A = np.array([[1, -p1], [1, -p3]])
    rhs = np.array([p2, p4])
    sol = np.linalg.solve(A, rhs)
    x = float(sol[1])
    y = float(sol[0])
    return (x, y)