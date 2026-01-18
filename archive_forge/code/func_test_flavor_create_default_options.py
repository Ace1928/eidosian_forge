from unittest import mock
from openstack.compute.v2 import flavor as _flavor
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import flavor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_flavor_create_default_options(self):
    arglist = [self.flavor.name]
    verifylist = [('name', self.flavor.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    default_args = {'name': self.flavor.name, 'ram': 256, 'vcpus': 1, 'disk': 0, 'id': None, 'ephemeral': 0, 'swap': 0, 'rxtx_factor': 1.0, 'is_public': True}
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.create_flavor.assert_called_once_with(**default_args)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)