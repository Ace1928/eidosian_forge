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
def test_flavor_unset_properties(self):
    arglist = ['--property', 'property1', '--property', 'property2', 'baremetal']
    verifylist = [('properties', ['property1', 'property2']), ('flavor', 'baremetal')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.compute_sdk_client.find_flavor.assert_called_with(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
    calls = [mock.call(self.flavor.id, 'property1'), mock.call(self.flavor.id, 'property2')]
    self.mock_shortcut.assert_has_calls(calls)
    calls.append(mock.call(self.flavor.id, 'property'))
    self.assertRaises(AssertionError, self.mock_shortcut.assert_has_calls, calls)
    self.compute_sdk_client.flavor_remove_tenant_access.assert_not_called()