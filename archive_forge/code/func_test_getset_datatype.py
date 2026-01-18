import collections.abc
import pickle
import pyomo.common.unittest as unittest
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.suffix import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.constraint import constraint, constraint_list
from pyomo.core.kernel.block import block, block_dict
def test_getset_datatype(self):
    s = suffix()
    s.set_datatype(suffix.FLOAT)
    self.assertEqual(s.get_datatype(), suffix.FLOAT)
    s.set_datatype(suffix.INT)
    self.assertEqual(s.get_datatype(), suffix.INT)
    with self.assertRaises(ValueError):
        s.set_datatype('something')