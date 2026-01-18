from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_get_param_value_missing(self):
    params = {'foo': 123}
    self.assertRaises(aws_exception.HeatMissingParameterError, api_utils.get_param_value, params, 'bar')