from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_list_validate_bad(self):
    schema = {'Type': 'CommaDelimitedList'}
    val_s = 0
    p = new_parameter('p', schema, validate_value=False)
    p.user_value = val_s
    err = self.assertRaises(exception.StackValidationFailed, p.validate)
    self.assertIn("Parameter 'p' is invalid", str(err))