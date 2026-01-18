from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_map_value_bad_parse(self):
    """Test value error for unparsable string value."""
    schema = {'Type': 'Json', 'ConstraintDescription': 'wibble'}
    val = 'I am not a map'
    err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, val)
    self.assertIn('Value must be valid JSON', str(err))