import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.network import Port, Arc
def test_fix_unfix(self):
    m = ConcreteModel()
    m.x = Var()
    m.port = Port()
    m.port.add(m.x)
    m.x.value = 10
    m.port.fix()
    self.assertTrue(m.x.is_fixed())
    m.port.unfix()
    self.assertFalse(m.x.is_fixed())