import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import (
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.variable import (
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import RealSet, IntegerSet, BooleanSet
from pyomo.core.base.set import (
def test_has_lb_ub(self):
    v = variable()
    self.assertEqual(v.has_lb(), False)
    self.assertEqual(v.lb, None)
    self.assertEqual(v.has_ub(), False)
    self.assertEqual(v.ub, None)
    v.lb = float('-inf')
    self.assertEqual(v.has_lb(), False)
    self.assertEqual(v.lb, None)
    self.assertEqual(v.has_ub(), False)
    self.assertEqual(v.ub, None)
    v.ub = float('inf')
    self.assertEqual(v.has_lb(), False)
    self.assertEqual(v.lb, None)
    self.assertEqual(v.has_ub(), False)
    self.assertEqual(v.ub, None)
    v.lb = 0
    self.assertEqual(v.has_lb(), True)
    self.assertEqual(v.lb, 0)
    self.assertEqual(v.has_ub(), False)
    self.assertEqual(v.ub, None)
    v.ub = 0
    self.assertEqual(v.has_lb(), True)
    self.assertEqual(v.lb, 0)
    self.assertEqual(v.has_ub(), True)
    self.assertEqual(v.ub, 0)
    v.lb = float('inf')
    self.assertEqual(v.has_lb(), True)
    self.assertEqual(v.lb, float('inf'))
    self.assertEqual(v.has_ub(), True)
    self.assertEqual(v.ub, 0)
    v.ub = float('-inf')
    self.assertEqual(v.has_lb(), True)
    self.assertEqual(v.lb, float('inf'))
    self.assertEqual(v.has_ub(), True)
    self.assertEqual(v.ub, float('-inf'))