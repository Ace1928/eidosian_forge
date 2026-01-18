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
def test_flavor_list_min_disk_min_ram(self):
    arglist = ['--min-disk', '10', '--min-ram', '2048']
    verifylist = [('min_disk', 10), ('min_ram', 2048)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'is_public': True, 'min_disk': 10, 'min_ram': 2048}
    self.compute_sdk_client.flavors.assert_called_with(**kwargs)
    self.compute_sdk_client.fetch_flavor_extra_specs.assert_not_called()
    self.assertEqual(self.columns, columns)
    self.assertEqual(tuple(self.data), tuple(data))