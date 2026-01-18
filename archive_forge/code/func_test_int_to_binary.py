import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Integers, value
from pyomo.environ import TransformationFactory as xfrm
from pyomo.common.log import LoggingIntercept
import logging
from io import StringIO
def test_int_to_binary(self):
    m = ConcreteModel()
    m.x = Var(domain=Integers, bounds=(0, 5))
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.preprocessing', logging.INFO):
        xfrm('contrib.integer_to_binary').apply_to(m)
    self.assertIn('Reformulating integer variables using the base2 strategy.', output.getvalue())
    reform_blk = m._int_to_binary_reform
    self.assertEqual(len(reform_blk.int_var_set), 1)
    reform_blk.new_binary_var[0, 0].value = 1
    reform_blk.new_binary_var[0, 1].value = 0
    reform_blk.new_binary_var[0, 2].value = 1
    m.x.value = 5
    self.assertEqual(value(reform_blk.integer_to_binary_constraint[0].body), 0)