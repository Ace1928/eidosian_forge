from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_format_response(self):
    response = api_utils.format_response('Foo', 'Bar')
    expected = {'FooResponse': {'FooResult': 'Bar'}}
    self.assertEqual(expected, response)