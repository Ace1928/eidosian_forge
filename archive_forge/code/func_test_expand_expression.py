import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_expand_expression(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.w = Var()
    m.prt1 = Port()
    m.prt1.add(-m.x, name='expr1')
    m.prt1.add(1 + m.y, name='expr2')
    m.prt2 = Port()
    m.prt2.add(-m.z, name='expr1')
    m.prt2.add(1 + m.w, name='expr2')
    m.c = Arc(ports=(m.prt1, m.prt2))
    self.assertEqual(len(list(m.component_objects(Constraint))), 0)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)
    TransformationFactory('network.expand_arcs').apply_to(m)
    self.assertEqual(len(list(m.component_objects(Constraint))), 2)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)
    self.assertFalse(m.c.active)
    blk = m.component('c_expanded')
    self.assertTrue(blk.active)
    self.assertTrue(blk.component('expr1_equality').active)
    self.assertTrue(blk.component('expr2_equality').active)
    os = StringIO()
    blk.pprint(ostream=os)
    self.assertEqual(os.getvalue(), 'c_expanded : Size=1, Index=None, Active=True\n    2 Constraint Declarations\n        expr1_equality : Size=1, Index=None, Active=True\n            Key  : Lower : Body    : Upper : Active\n            None :   0.0 : - x + z :   0.0 :   True\n        expr2_equality : Size=1, Index=None, Active=True\n            Key  : Lower : Body            : Upper : Active\n            None :   0.0 : 1 + y - (1 + w) :   0.0 :   True\n\n    2 Declarations: expr1_equality expr2_equality\n')