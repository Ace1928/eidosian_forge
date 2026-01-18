import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_extensive_single_var(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.p1 = Port(initialize={'v': (m.x, Port.Extensive)})
    m.p2 = Port(initialize={'v': (m.y, Port.Extensive)})
    m.p3 = Port(initialize={'v': (m.z, Port.Extensive)})
    m.a1 = Arc(source=m.p1, destination=m.p2)
    m.a2 = Arc(source=m.p1, destination=m.p3)
    TransformationFactory('network.expand_arcs').apply_to(m)
    os = StringIO()
    m.pprint(ostream=os)
    self.assertEqual(os.getvalue(), '3 Var Declarations\n    x : Size=1, Index=None\n        Key  : Lower : Value : Upper : Fixed : Stale : Domain\n        None :  None :  None :  None : False :  True :  Reals\n    y : Size=1, Index=None\n        Key  : Lower : Value : Upper : Fixed : Stale : Domain\n        None :  None :  None :  None : False :  True :  Reals\n    z : Size=1, Index=None\n        Key  : Lower : Value : Upper : Fixed : Stale : Domain\n        None :  None :  None :  None : False :  True :  Reals\n\n3 Constraint Declarations\n    p1_v_outsum : Size=1, Index=None, Active=True\n        Key  : Lower : Body                              : Upper : Active\n        None :   0.0 : a1_expanded.v + a2_expanded.v - x :   0.0 :   True\n    p2_v_insum : Size=1, Index=None, Active=True\n        Key  : Lower : Body              : Upper : Active\n        None :   0.0 : a1_expanded.v - y :   0.0 :   True\n    p3_v_insum : Size=1, Index=None, Active=True\n        Key  : Lower : Body              : Upper : Active\n        None :   0.0 : a2_expanded.v - z :   0.0 :   True\n\n2 Block Declarations\n    a1_expanded : Size=1, Index=None, Active=True\n        1 Var Declarations\n            v : Size=1, Index=None\n                Key  : Lower : Value : Upper : Fixed : Stale : Domain\n                None :  None :  None :  None : False :  True :  Reals\n\n        1 Declarations: v\n    a2_expanded : Size=1, Index=None, Active=True\n        1 Var Declarations\n            v : Size=1, Index=None\n                Key  : Lower : Value : Upper : Fixed : Stale : Domain\n                None :  None :  None :  None : False :  True :  Reals\n\n        1 Declarations: v\n\n2 Arc Declarations\n    a1 : Size=1, Index=None, Active=False\n        Key  : Ports    : Directed : Active\n        None : (p1, p2) :     True :  False\n    a2 : Size=1, Index=None, Active=False\n        Key  : Ports    : Directed : Active\n        None : (p1, p3) :     True :  False\n\n3 Port Declarations\n    p1 : Size=1, Index=None\n        Key  : Name : Size : Variable\n        None :    v :    1 :        x\n    p2 : Size=1, Index=None\n        Key  : Name : Size : Variable\n        None :    v :    1 :        y\n    p3 : Size=1, Index=None\n        Key  : Name : Size : Variable\n        None :    v :    1 :        z\n\n13 Declarations: x y z p1 p2 p3 a1 a2 a1_expanded a2_expanded p1_v_outsum p2_v_insum p3_v_insum\n')