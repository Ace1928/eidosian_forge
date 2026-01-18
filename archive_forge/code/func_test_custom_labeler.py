import pyomo.common.unittest as unittest
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.variable import variable
from pyomo.environ import ConcreteModel, Var
def test_custom_labeler(self):
    labeler = lambda x, y: '^' + str(x) + y
    s = SymbolMap(lambda x: '_' + str(x))
    v = variable()
    self.assertEqual('^' + str(v) + '~', s.getSymbol(v, labeler, '~'))
    s = SymbolMap(lambda x: '_' + str(x))
    m = ConcreteModel()
    m.x = Var()
    self.assertEqual('^x~', s.createSymbol(m.x, labeler, '~'))
    s = SymbolMap(lambda x: '_' + str(x))
    m.y = Var([1, 2, 3])
    s.createSymbols(m.y.values(), labeler, '~')
    self.assertEqual(s.bySymbol, {'^y[1]~': m.y[1], '^y[2]~': m.y[2], '^y[3]~': m.y[3]})
    self.assertEqual(s.byObject, {id(m.y[1]): '^y[1]~', id(m.y[2]): '^y[2]~', id(m.y[3]): '^y[3]~'})