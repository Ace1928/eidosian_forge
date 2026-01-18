import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
@unittest.skipUnless(pyo.SolverFactory('ipopt').available(), 'Ipopt is not available')
def test_fix_then_solve(self):
    m = _make_simple_model()
    ipopt = pyo.SolverFactory('ipopt')
    m.v1.set_value(1.0)
    m.v2.set_value(1.0)
    m.v3.set_value(1.0)
    m.v4.set_value(2.0)
    with TemporarySubsystemManager(to_fix=[m.v3, m.v4], to_deactivate=[m.con1]):
        ipopt.solve(m)
    self.assertAlmostEqual(m.v1.value, pyo.sqrt(7.0), delta=1e-08)
    self.assertAlmostEqual(m.v2.value, pyo.sqrt(4.0 - pyo.sqrt(7.0)), delta=1e-08)