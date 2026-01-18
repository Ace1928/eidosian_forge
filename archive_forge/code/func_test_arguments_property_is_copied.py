from unittest import mock
import novaclient
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient import utils as nutils
from novaclient.v2 import versions
@mock.patch('novaclient.api_versions._get_function_name')
def test_arguments_property_is_copied(self, mock_name):

    @nutils.arg('argument_1')
    @api_versions.wraps('2.666', '2.777')
    @nutils.arg('argument_2')
    def some_func():
        pass
    versioned_method = api_versions.get_substitutions(mock_name.return_value, api_versions.APIVersion('2.700'))[0]
    self.assertEqual(some_func.arguments, versioned_method.func.arguments)
    self.assertIn((('argument_1',), {}), versioned_method.func.arguments)
    self.assertIn((('argument_2',), {}), versioned_method.func.arguments)