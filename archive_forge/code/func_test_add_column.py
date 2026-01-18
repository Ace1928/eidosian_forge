import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.solvers.plugins.solvers.xpress_direct import xpress_available
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
@unittest.skipIf(not xpress_available, 'xpress is not available')
def test_add_column(self):
    m = pe.ConcreteModel()
    m.x = pe.Var(within=pe.NonNegativeReals)
    m.c = pe.Constraint(expr=(0, m.x, 1))
    m.obj = pe.Objective(expr=-m.x)
    opt = pe.SolverFactory('xpress_persistent')
    opt.set_instance(m)
    opt.solve()
    self.assertAlmostEqual(m.x.value, 1)
    m.y = pe.Var(within=pe.NonNegativeReals)
    opt.add_column(m, m.y, -3, [m.c], [2])
    opt.solve()
    self.assertAlmostEqual(m.x.value, 0)
    self.assertAlmostEqual(m.y.value, 0.5)