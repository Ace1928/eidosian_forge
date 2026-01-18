import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_expand_single_scalar(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.prt1 = Port()
    m.prt1.add(m.x, 'v')
    m.prt2 = Port()
    m.prt2.add(m.y, 'v')
    m.c = Arc(source=m.prt1, destination=m.prt2)
    m.d = Arc()
    self.assertEqual(len(list(m.component_objects(Constraint))), 0)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)
    TransformationFactory('network.expand_arcs').apply_to(m)
    self.assertEqual(len(list(m.component_objects(Constraint))), 1)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 1)
    self.assertFalse(m.c.active)
    blk = m.component('c_expanded')
    self.assertTrue(blk.active)
    self.assertTrue(blk.component('v_equality').active)
    self.assertTrue(m.d_expanded.active)
    self.assertEqual(len(list(m.d_expanded.component_objects())), 0)
    os = StringIO()
    blk.pprint(ostream=os)
    self.assertEqual(os.getvalue(), 'c_expanded : Size=1, Index=None, Active=True\n    1 Constraint Declarations\n        v_equality : Size=1, Index=None, Active=True\n            Key  : Lower : Body  : Upper : Active\n            None :   0.0 : x - y :   0.0 :   True\n\n    1 Declarations: v_equality\n')