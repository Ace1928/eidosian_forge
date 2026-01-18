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
def test_fix_free(self):
    v = variable()
    self.assertEqual(v.value, None)
    self.assertEqual(v.fixed, False)
    v.fix(1)
    self.assertEqual(v.value, 1)
    self.assertEqual(v.fixed, True)
    v.free()
    self.assertEqual(v.value, 1)
    self.assertEqual(v.fixed, False)
    v.value = 0
    self.assertEqual(v.value, 0)
    self.assertEqual(v.fixed, False)
    v.fix()
    self.assertEqual(v.value, 0)
    self.assertEqual(v.fixed, True)
    with self.assertRaises(TypeError):
        v.fix(1, 2)
    self.assertEqual(v.value, 0)
    self.assertEqual(v.fixed, True)
    v.free()
    with self.assertRaises(TypeError):
        v.fix(1, 2)
    self.assertEqual(v.value, 0)
    self.assertEqual(v.fixed, False)