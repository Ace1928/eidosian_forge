import pyomo.common.unittest as unittest
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.variable import variable
from pyomo.environ import ConcreteModel, Var
def test_add_symbol(self):
    s = SymbolMap()
    m = ConcreteModel()
    m.x = Var()
    m.y = Var([1, 2, 3])
    s.addSymbol(m.x, 'x')
    self.assertEqual(s.bySymbol, {'x': m.x})
    self.assertEqual(s.byObject, {id(m.x): 'x'})
    with self.assertRaisesRegex(RuntimeError, 'SymbolMap.addSymbol\\(\\): duplicate symbol.'):
        s.addSymbol(m.y, 'x')
    s = SymbolMap()
    s.addSymbol(m.x, 'x')
    with self.assertRaisesRegex(RuntimeError, 'SymbolMap.addSymbol\\(\\): duplicate object.'):
        s.addSymbol(m.x, 'y')