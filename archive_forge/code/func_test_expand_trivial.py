import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_expand_trivial(self):
    m = ConcreteModel()
    m.x = Var()
    m.prt = Port()
    m.prt.add(m.x, 'a')
    m.c = Arc(ports=(m.prt, m.prt))
    self.assertEqual(len(list(m.component_objects(Constraint))), 0)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)
    TransformationFactory('network.expand_arcs').apply_to(m)
    self.assertEqual(len(list(m.component_objects(Constraint))), 1)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 1)
    self.assertFalse(m.c.active)
    blk = m.component('c_expanded')
    self.assertTrue(blk.active)
    self.assertTrue(blk.component('a_equality').active)
    os = StringIO()
    blk.pprint(ostream=os)
    self.assertEqual(os.getvalue(), 'c_expanded : Size=1, Index=None, Active=True\n    1 Constraint Declarations\n        a_equality : Size=1, Index=None, Active=True\n            Key  : Lower : Body  : Upper : Active\n            None :   0.0 : x - x :   0.0 :   True\n\n    1 Declarations: a_equality\n')