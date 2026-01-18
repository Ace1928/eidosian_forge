import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.network import Port, Arc
def test_display(self):
    pipe = ConcreteModel()
    pipe.SPECIES = Set(initialize=['a', 'b', 'c'])
    pipe.flow = Var(initialize=10)
    pipe.composition = Var(pipe.SPECIES, initialize=lambda m, i: ord(i) - ord('a'))
    pipe.pIn = Var(within=NonNegativeReals, initialize=3.14)
    pipe.OUT = Port(implicit=['imp'])
    pipe.OUT.add(-pipe.flow, 'flow')
    pipe.OUT.add(pipe.composition, 'composition')
    pipe.OUT.add(pipe.pIn, 'pressure')
    os = StringIO()
    pipe.OUT.display(ostream=os)
    self.assertEqual(os.getvalue(), "OUT : Size=1\n    Key  : Name        : Value\n    None : composition : {'a': 0, 'b': 1, 'c': 2}\n         :        flow : -10\n         :         imp : -\n         :    pressure : 3.14\n")

    def _IN(m, i):
        return {'pressure': pipe.pIn, 'flow': pipe.composition[i] * pipe.flow}
    pipe.IN = Port(pipe.SPECIES, rule=_IN)
    os = StringIO()
    pipe.IN.display(ostream=os)
    self.assertEqual(os.getvalue(), 'IN : Size=3\n    Key : Name     : Value\n      a :     flow :     0\n        : pressure :  3.14\n      b :     flow :    10\n        : pressure :  3.14\n      c :     flow :    20\n        : pressure :  3.14\n')