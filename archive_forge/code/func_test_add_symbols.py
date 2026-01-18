import pyomo.common.unittest as unittest
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.variable import variable
from pyomo.environ import ConcreteModel, Var
def test_add_symbols(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var([1, 2, 3])
    s = SymbolMap()
    s.addSymbols(((m.y[i], str(i)) for i in (1, 2, 3)))
    self.assertEqual(s.bySymbol, {'1': m.y[1], '2': m.y[2], '3': m.y[3]})
    self.assertEqual(s.byObject, {id(m.y[1]): '1', id(m.y[2]): '2', id(m.y[3]): '3'})
    with self.assertRaisesRegex(RuntimeError, 'SymbolMap.addSymbols\\(\\): duplicate symbol.'):
        s.addSymbols([(m.y, '1')])
    s = SymbolMap()
    s.addSymbols(((m.y[i], str(i)) for i in (1, 2, 3)))
    with self.assertRaisesRegex(RuntimeError, 'SymbolMap.addSymbols\\(\\): duplicate object.'):
        s.addSymbols([(m.y[2], 'x')])