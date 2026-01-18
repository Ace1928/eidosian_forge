from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_missing_params(self):
    user_params = {}
    self.assertRaises(exception.UserParameterMissing, self.new_parameters, 'test', params_schema, user_params)