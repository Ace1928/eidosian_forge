import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.solvers.plugins.solvers.xpress_direct import xpress_available
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
@unittest.skipIf(not xpress_available, 'xpress is not available')
def test_add_remove_var(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    opt = pe.SolverFactory('xpress_persistent')
    opt.set_instance(m)
    self.assertEqual(opt.get_xpress_attribute('cols'), 2)
    opt.remove_var(m.x)
    self.assertEqual(opt.get_xpress_attribute('cols'), 1)
    opt.add_var(m.x)
    self.assertEqual(opt.get_xpress_attribute('cols'), 2)
    opt.remove_var(m.x)
    opt.add_var(m.x)
    opt.remove_var(m.x)
    self.assertEqual(opt.get_xpress_attribute('cols'), 1)