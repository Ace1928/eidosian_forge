from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_bool_value_invalid(self):
    schema = {'Type': 'Boolean'}
    err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'bo', schema, 'foo')
    self.assertIn("Unrecognized value 'foo'", str(err))