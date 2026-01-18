import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_expand_empty_expression(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.PRT = Port()
    m.PRT.add(-m.x, 'x')
    m.PRT.add(1 + m.y, 'y')
    m.EPRT = Port()
    m.c = Arc(ports=(m.PRT, m.EPRT))
    self.assertEqual(len(list(m.component_objects(Constraint))), 0)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)
    TransformationFactory('network.expand_arcs').apply_to(m)
    self.assertEqual(len(list(m.component_objects(Constraint))), 2)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)
    self.assertFalse(m.c.active)
    blk = m.component('c_expanded')
    self.assertTrue(blk.component('x_equality').active)
    self.assertTrue(blk.component('y_equality').active)
    os = StringIO()
    blk.pprint(ostream=os)
    self.assertEqual(os.getvalue(), 'c_expanded : Size=1, Index=None, Active=True\n    2 Constraint Declarations\n        x_equality : Size=1, Index=None, Active=True\n            Key  : Lower : Body              : Upper : Active\n            None :   0.0 : - x - EPRT_auto_x :   0.0 :   True\n        y_equality : Size=1, Index=None, Active=True\n            Key  : Lower : Body                : Upper : Active\n            None :   0.0 : 1 + y - EPRT_auto_y :   0.0 :   True\n\n    2 Declarations: x_equality y_equality\n')