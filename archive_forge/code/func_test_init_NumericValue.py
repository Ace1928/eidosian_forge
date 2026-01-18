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
def test_init_NumericValue(self):
    v = variable()
    p = parameter()
    e = expression()
    d = data_expression()
    o = objective()
    for obj in (v, v + 1, v ** 2, p, p + 1, p ** 2, e, e + 1, e ** 2, d, d + 1, d ** 2, o, o + 1, o ** 2):
        self.assertTrue(isinstance(noclone(obj), NumericValue))
        self.assertTrue(isinstance(noclone(obj), IIdentityExpression))
        self.assertIs(noclone(obj).expr, obj)