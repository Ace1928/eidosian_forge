from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_map_underrun(self):
    """Test map length under MIN_LEN."""
    schema = {'Type': 'Json', 'MinLength': 3}
    val = {'foo': 'bar', 'items': [1, 2, 3]}
    err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, val)
    self.assertIn('out of range', str(err))