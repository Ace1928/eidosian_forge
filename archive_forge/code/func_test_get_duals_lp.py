import pyomo.common.unittest as unittest
from pyomo.opt import TerminationCondition, SolutionStatus, check_available_solvers
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
def test_get_duals_lp(self):
    model = pyo.ConcreteModel()
    model.X = pyo.Var(within=pyo.NonNegativeReals)
    model.Y = pyo.Var(within=pyo.NonNegativeReals)
    model.C1 = pyo.Constraint(expr=2 * model.X + model.Y >= 8)
    model.C2 = pyo.Constraint(expr=model.X + 3 * model.Y >= 6)
    model.O = pyo.Objective(expr=model.X + model.Y)
    opt = pyo.SolverFactory('mosek_direct')
    results = opt.solve(model, suffixes=['dual'], load_solutions=False)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.solutions.load_from(results)
    self.assertAlmostEqual(model.dual[model.C1], 0.4, 4)
    self.assertAlmostEqual(model.dual[model.C2], 0.2, 4)