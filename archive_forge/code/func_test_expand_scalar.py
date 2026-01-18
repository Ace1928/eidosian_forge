import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_expand_scalar(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.w = Var()
    m.prt1 = Port()
    m.prt1.add(m.x, 'a')
    m.prt1.add(m.y, 'b')
    m.prt2 = Port()
    m.prt2.add(m.z, 'a')
    m.prt2.add(m.w, 'b')
    m.c = Arc(ports=(m.prt1, m.prt2))
    self.assertEqual(len(list(m.component_objects(Constraint))), 0)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)
    TransformationFactory('network.expand_arcs').apply_to(m)
    self.assertEqual(len(list(m.component_objects(Constraint))), 2)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)
    self.assertFalse(m.c.active)
    blk = m.component('c_expanded')
    self.assertTrue(blk.active)
    self.assertTrue(blk.component('a_equality').active)
    self.assertTrue(blk.component('b_equality').active)
    os = StringIO()
    blk.pprint(ostream=os)
    self.assertEqual(os.getvalue(), 'c_expanded : Size=1, Index=None, Active=True\n    2 Constraint Declarations\n        a_equality : Size=1, Index=None, Active=True\n            Key  : Lower : Body  : Upper : Active\n            None :   0.0 : x - z :   0.0 :   True\n        b_equality : Size=1, Index=None, Active=True\n            Key  : Lower : Body  : Upper : Active\n            None :   0.0 : y - w :   0.0 :   True\n\n    2 Declarations: a_equality b_equality\n')