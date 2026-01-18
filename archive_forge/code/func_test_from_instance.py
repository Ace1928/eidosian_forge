import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.symbol_map import symbol_map_from_instance
def test_from_instance(self):
    smap = symbol_map_from_instance(self.instance)
    self.assertEqual(set(smap.bySymbol.keys()), set(['x', 'y(1)', 'y(2)', 'y(3)', 'b_x', 'b_y(1)', 'b_y(2)', 'b_y(3)', 'o1', 'o2(1)', 'o2(2)', 'o2(3)', 'c1', 'c2(1)', 'c2(2)', 'c2(3)']))
    self.assertEqual(set(smap.aliases.keys()), set(['c_e_c2(3)_', '__default_objective__', 'c_u_c2(1)_', 'c_l_c1_', 'r_u_c2(2)_', 'r_l_c2(2)_']))