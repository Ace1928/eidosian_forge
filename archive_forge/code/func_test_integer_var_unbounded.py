import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Integers, value
from pyomo.environ import TransformationFactory as xfrm
from pyomo.common.log import LoggingIntercept
import logging
from io import StringIO
def test_integer_var_unbounded(self):
    m = ConcreteModel()
    m.x = Var(domain=Integers)
    with self.assertRaises(ValueError):
        xfrm('contrib.integer_to_binary').apply_to(m)