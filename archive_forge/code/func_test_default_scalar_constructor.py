import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_default_scalar_constructor(self):
    m = ConcreteModel()
    m.c1 = Arc()
    self.assertEqual(len(m.c1), 0)
    self.assertIsNone(m.c1.directed)
    self.assertIsNone(m.c1.ports)
    self.assertIsNone(m.c1.source)
    self.assertIsNone(m.c1.destination)
    m = AbstractModel()
    m.c1 = Arc()
    self.assertEqual(len(m.c1), 0)
    self.assertIsNone(m.c1.directed)
    self.assertIsNone(m.c1.ports)
    self.assertIsNone(m.c1.source)
    self.assertIsNone(m.c1.destination)
    inst = m.create_instance()
    self.assertEqual(len(inst.c1), 0)
    self.assertIsNone(inst.c1.directed)
    self.assertIsNone(inst.c1.ports)
    self.assertIsNone(inst.c1.source)
    self.assertIsNone(inst.c1.destination)