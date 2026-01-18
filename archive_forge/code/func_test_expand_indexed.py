import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_expand_indexed(self):
    m = ConcreteModel()
    m.x = Var([1, 2])
    m.y = Var([1, 2], [1, 2])
    m.z = Var()
    m.t = Var([1, 2])
    m.u = Var([1, 2], [1, 2])
    m.v = Var()
    m.prt1 = Port()
    m.prt1.add(m.x, 'a')
    m.prt1.add(m.y, 'b')
    m.prt1.add(m.z, 'c')
    m.prt2 = Port()
    m.prt2.add(m.t, 'a')
    m.prt2.add(m.u, 'b')
    m.prt2.add(m.v, 'c')
    m.c = Arc(ports=(m.prt1, m.prt2))
    self.assertEqual(len(list(m.component_objects(Constraint))), 0)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)
    TransformationFactory('network.expand_arcs').apply_to(m)
    self.assertEqual(len(list(m.component_objects(Constraint))), 3)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 7)
    self.assertFalse(m.c.active)
    blk = m.component('c_expanded')
    self.assertTrue(blk.active)
    self.assertTrue(blk.component('a_equality').active)
    os = StringIO()
    blk.pprint(ostream=os)
    self.assertEqual(os.getvalue(), 'c_expanded : Size=1, Index=None, Active=True\n    3 Constraint Declarations\n        a_equality : Size=2, Index={1, 2}, Active=True\n            Key : Lower : Body        : Upper : Active\n              1 :   0.0 : x[1] - t[1] :   0.0 :   True\n              2 :   0.0 : x[2] - t[2] :   0.0 :   True\n        b_equality : Size=4, Index={1, 2}*{1, 2}, Active=True\n            Key    : Lower : Body            : Upper : Active\n            (1, 1) :   0.0 : y[1,1] - u[1,1] :   0.0 :   True\n            (1, 2) :   0.0 : y[1,2] - u[1,2] :   0.0 :   True\n            (2, 1) :   0.0 : y[2,1] - u[2,1] :   0.0 :   True\n            (2, 2) :   0.0 : y[2,2] - u[2,2] :   0.0 :   True\n        c_equality : Size=1, Index=None, Active=True\n            Key  : Lower : Body  : Upper : Active\n            None :   0.0 : z - v :   0.0 :   True\n\n    3 Declarations: a_equality b_equality c_equality\n')