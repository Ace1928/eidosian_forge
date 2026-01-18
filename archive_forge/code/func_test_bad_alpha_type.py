import pickle
import math
import pyomo.common.unittest as unittest
from pyomo.kernel import pprint, IntegerSet
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.variable import variable, variable_tuple
from pyomo.core.kernel.block import block
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression, data_expression
from pyomo.core.kernel.conic import (
def test_bad_alpha_type(self):
    c = dual_power(r1=variable(lb=0), r2=variable(lb=0), x=[variable(), variable()], alpha=parameter())
    c = dual_power(r1=variable(lb=0), r2=variable(lb=0), x=[variable(), variable()], alpha=data_expression())
    with self.assertRaises(TypeError):
        c = dual_power(r1=variable(lb=0), r2=variable(lb=0), x=[variable(), variable()], alpha=variable())
    with self.assertRaises(TypeError):
        c = dual_power(r1=variable(lb=0), r2=variable(lb=0), x=[variable(), variable()], alpha=expression())