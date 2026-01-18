import collections.abc
import pickle
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.core.kernel.variable import variable, variable_dict, variable_list
from pyomo.core.kernel.constraint import constraint, constraint_dict, constraint_list
from pyomo.core.kernel.objective import objective, objective_dict, objective_list
from pyomo.core.kernel.expression import expression, expression_dict, expression_list
from pyomo.core.kernel.block import block, block_dict, block_list
from pyomo.core.kernel.suffix import suffix
def test_misc_set_ops(self):
    v1 = variable()
    cset1 = ComponentSet([v1])
    v2 = variable()
    cset2 = ComponentSet([v2])
    cset3 = ComponentSet([v1, v2])
    empty = ComponentSet([])
    self.assertEqual(cset1 | cset2, cset3)
    self.assertEqual((cset1 | cset2) - cset3, empty)
    self.assertEqual(cset1 ^ cset2, cset3)
    self.assertEqual(cset1 ^ cset3, cset2)
    self.assertEqual(cset2 ^ cset3, cset1)
    self.assertEqual(cset1 & cset2, empty)
    self.assertEqual(cset1 & cset3, cset1)
    self.assertEqual(cset2 & cset3, cset2)