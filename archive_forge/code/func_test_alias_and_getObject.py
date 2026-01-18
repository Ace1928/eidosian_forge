import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.symbol_map import symbol_map_from_instance
def test_alias_and_getObject(self):
    smap = SymbolMap()
    smap.addSymbol(self.instance.x, 'x')
    smap.alias(self.instance.x, 'X')
    self.assertEqual(set(smap.bySymbol.keys()), set(['x']))
    self.assertEqual(set(smap.aliases.keys()), set(['X']))
    self.assertEqual(id(smap.getObject('x')), id(self.instance.x))
    self.assertEqual(id(smap.getObject('X')), id(self.instance.x))