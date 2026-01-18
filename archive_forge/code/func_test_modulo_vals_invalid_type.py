from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_modulo_vals_invalid_type(self):
    self.assertRaises(exception.InvalidSchemaError, constraints.Modulo, '2', 1)
    self.assertRaises(exception.InvalidSchemaError, constraints.Modulo, 2, '1')