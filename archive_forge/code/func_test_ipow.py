import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import (
import pyomo.kernel
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.expression import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.objective import objective
from pyomo.core.kernel.block import block
def test_ipow(self):
    e = self._ctype_factory(3.0)
    expr = e
    for v in [2.0, 1.0]:
        expr **= v
    self.assertEqual(e.expr, 3)
    self.assertEqual(expr(), 9)
    expr = e
    for v in [1.0, 2.0]:
        expr **= v
    self.assertEqual(e.expr, 3)
    self.assertEqual(expr(), 9)