from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_allowed_pattern_invalid_type(self):
    schema = constraints.Schema('Integer', constraints=[constraints.AllowedPattern('[0-9]*')])
    err = self.assertRaises(exception.InvalidSchemaError, schema.validate)
    self.assertIn('AllowedPattern constraint invalid for Integer', str(err))