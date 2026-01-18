from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Objective, Var, log10, minimize
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases
def test_BARON_option_warnings(self):
    os = StringIO()
    with LoggingIntercept(os, 'pyomo.solvers'):
        m = ConcreteModel()
        m.x = Var()
        m.obj = Objective(expr=m.x ** 2)
        with SolverFactory('baron') as opt:
            results = opt.solve(m, options={'ResName': 'results.lst', 'TimName': 'results.tim'})
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertIn('Ignoring user-specified option "ResName=results.lst"', os.getvalue())
    self.assertIn('Ignoring user-specified option "TimName=results.tim"', os.getvalue())