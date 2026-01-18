import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.solvers.plugins.solvers.xpress_direct import xpress_available
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
@unittest.skipIf(not xpress_available, 'xpress is not available')
def test_add_remove_sosconstraint2(self):
    m = pe.ConcreteModel()
    m.a = pe.Set(initialize=[1, 2, 3], ordered=True)
    m.x = pe.Var(m.a, within=pe.Binary)
    m.y = pe.Var(within=pe.Binary)
    m.obj = pe.Objective(expr=m.y)
    m.c1 = pe.SOSConstraint(var=m.x, sos=1)
    opt = pe.SolverFactory('xpress_persistent')
    opt.set_instance(m)
    self.assertEqual(opt.get_xpress_attribute('sets'), 1)
    m.c2 = pe.SOSConstraint(var=m.x, sos=2)
    opt.add_sos_constraint(m.c2)
    self.assertEqual(opt.get_xpress_attribute('sets'), 2)
    opt.remove_sos_constraint(m.c2)
    self.assertEqual(opt.get_xpress_attribute('sets'), 1)