from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_missing_param_map(self):
    """Test missing user parameter."""
    self.assertRaises(exception.UserParameterMissing, new_parameter, 'p', {'Type': 'Json'})