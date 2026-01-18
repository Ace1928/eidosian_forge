import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_extensive_no_splitfrac_expansion(self):
    m = ConcreteModel()
    m.time = Set(initialize=[1, 2, 3])
    m.source = Block()
    m.load1 = Block()
    m.load2 = Block()

    def source_block(b):
        b.p_out = Var(b.model().time)
        b.outlet = Port(initialize={'p': (b.p_out, Port.Extensive, {'include_splitfrac': False})})

    def load_block(b):
        b.p_in = Var(b.model().time)
        b.inlet = Port(initialize={'p': (b.p_in, Port.Extensive, {'include_splitfrac': False})})
    source_block(m.source)
    load_block(m.load1)
    load_block(m.load2)
    m.cs1 = Arc(source=m.source.outlet, destination=m.load1.inlet)
    m.cs2 = Arc(source=m.source.outlet, destination=m.load2.inlet)
    TransformationFactory('network.expand_arcs').apply_to(m)
    ref = '\n1 Set Declarations\n    time : Size=1, Index=None, Ordered=Insertion\n        Key  : Dimen : Domain : Size : Members\n        None :     1 :    Any :    3 : {1, 2, 3}\n\n5 Block Declarations\n    cs1_expanded : Size=1, Index=None, Active=True\n        1 Var Declarations\n            p : Size=3, Index=time\n                Key : Lower : Value : Upper : Fixed : Stale : Domain\n                  1 :  None :  None :  None : False :  True :  Reals\n                  2 :  None :  None :  None : False :  True :  Reals\n                  3 :  None :  None :  None : False :  True :  Reals\n\n        1 Declarations: p\n    cs2_expanded : Size=1, Index=None, Active=True\n        1 Var Declarations\n            p : Size=3, Index=time\n                Key : Lower : Value : Upper : Fixed : Stale : Domain\n                  1 :  None :  None :  None : False :  True :  Reals\n                  2 :  None :  None :  None : False :  True :  Reals\n                  3 :  None :  None :  None : False :  True :  Reals\n\n        1 Declarations: p\n    load1 : Size=1, Index=None, Active=True\n        1 Var Declarations\n            p_in : Size=3, Index=time\n                Key : Lower : Value : Upper : Fixed : Stale : Domain\n                  1 :  None :  None :  None : False :  True :  Reals\n                  2 :  None :  None :  None : False :  True :  Reals\n                  3 :  None :  None :  None : False :  True :  Reals\n\n        1 Constraint Declarations\n            inlet_p_insum : Size=3, Index=time, Active=True\n                Key : Lower : Body                              : Upper : Active\n                  1 :   0.0 : cs1_expanded.p[1] - load1.p_in[1] :   0.0 :   True\n                  2 :   0.0 : cs1_expanded.p[2] - load1.p_in[2] :   0.0 :   True\n                  3 :   0.0 : cs1_expanded.p[3] - load1.p_in[3] :   0.0 :   True\n\n        1 Port Declarations\n            inlet : Size=1, Index=None\n                Key  : Name : Size : Variable\n                None :    p :    3 : load1.p_in\n\n        3 Declarations: p_in inlet inlet_p_insum\n    load2 : Size=1, Index=None, Active=True\n        1 Var Declarations\n            p_in : Size=3, Index=time\n                Key : Lower : Value : Upper : Fixed : Stale : Domain\n                  1 :  None :  None :  None : False :  True :  Reals\n                  2 :  None :  None :  None : False :  True :  Reals\n                  3 :  None :  None :  None : False :  True :  Reals\n\n        1 Constraint Declarations\n            inlet_p_insum : Size=3, Index=time, Active=True\n                Key : Lower : Body                              : Upper : Active\n                  1 :   0.0 : cs2_expanded.p[1] - load2.p_in[1] :   0.0 :   True\n                  2 :   0.0 : cs2_expanded.p[2] - load2.p_in[2] :   0.0 :   True\n                  3 :   0.0 : cs2_expanded.p[3] - load2.p_in[3] :   0.0 :   True\n\n        1 Port Declarations\n            inlet : Size=1, Index=None\n                Key  : Name : Size : Variable\n                None :    p :    3 : load2.p_in\n\n        3 Declarations: p_in inlet inlet_p_insum\n    source : Size=1, Index=None, Active=True\n        1 Var Declarations\n            p_out : Size=3, Index=time\n                Key : Lower : Value : Upper : Fixed : Stale : Domain\n                  1 :  None :  None :  None : False :  True :  Reals\n                  2 :  None :  None :  None : False :  True :  Reals\n                  3 :  None :  None :  None : False :  True :  Reals\n\n        1 Constraint Declarations\n            outlet_p_outsum : Size=3, Index=time, Active=True\n                Key : Lower : Body                                                    : Upper : Active\n                  1 :   0.0 : cs1_expanded.p[1] + cs2_expanded.p[1] - source.p_out[1] :   0.0 :   True\n                  2 :   0.0 : cs1_expanded.p[2] + cs2_expanded.p[2] - source.p_out[2] :   0.0 :   True\n                  3 :   0.0 : cs1_expanded.p[3] + cs2_expanded.p[3] - source.p_out[3] :   0.0 :   True\n\n        1 Port Declarations\n            outlet : Size=1, Index=None\n                Key  : Name : Size : Variable\n                None :    p :    3 : source.p_out\n\n        3 Declarations: p_out outlet outlet_p_outsum\n\n2 Arc Declarations\n    cs1 : Size=1, Index=None, Active=False\n        Key  : Ports                        : Directed : Active\n        None : (source.outlet, load1.inlet) :     True :  False\n    cs2 : Size=1, Index=None, Active=False\n        Key  : Ports                        : Directed : Active\n        None : (source.outlet, load2.inlet) :     True :  False\n\n8 Declarations: time source load1 load2 cs1 cs2 cs1_expanded cs2_expanded\n'
    os = StringIO()
    m.pprint(ostream=os)
    self.assertEqual(os.getvalue().strip(), ref.strip())