import pickle
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import dill, dill_available as has_dill
from pyomo.core.expr.numvalue import (
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.parameter import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.block import block
@unittest.skipIf(not has_dill, 'The dill module is not available')
def test_dill(self):
    p = parameter(1)
    f = functional_value(lambda: p())
    self.assertEqual(f(), 1)
    fup = dill.loads(dill.dumps(f))
    p.value = 2
    self.assertEqual(f(), 2)
    self.assertEqual(fup(), 1)
    b = block()
    b.p = p
    b.f = f
    self.assertEqual(b.f(), 2)
    bup = dill.loads(dill.dumps(b))
    fup = bup.f
    b.p.value = 4
    self.assertEqual(b.f(), 4)
    self.assertEqual(bup.f(), 2)
    bup.p.value = 4
    self.assertEqual(bup.f(), 4)