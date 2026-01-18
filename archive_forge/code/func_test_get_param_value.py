from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_get_param_value(self):
    params = {'foo': 123}
    self.assertEqual(123, api_utils.get_param_value(params, 'foo'))