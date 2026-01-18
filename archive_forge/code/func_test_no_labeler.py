import pyomo.common.unittest as unittest
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.variable import variable
from pyomo.environ import ConcreteModel, Var
def test_no_labeler(self):
    s = SymbolMap()
    v = variable()
    self.assertEqual(str(v), s.getSymbol(v))
    s = SymbolMap()
    m = ConcreteModel()
    m.x = Var()
    self.assertEqual('x', s.createSymbol(m.x))
    s = SymbolMap()
    m.y = Var([1, 2, 3])
    s.createSymbols(m.y.values())
    self.assertEqual(s.bySymbol, {'y[1]': m.y[1], 'y[2]': m.y[2], 'y[3]': m.y[3]})
    self.assertEqual(s.byObject, {id(m.y[1]): 'y[1]', id(m.y[2]): 'y[2]', id(m.y[3]): 'y[3]'})