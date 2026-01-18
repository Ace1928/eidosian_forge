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
def test_multi_flavors_delete_with_exception(self):
    arglist = [self.flavors[0].id, 'unexist_flavor']
    verifylist = [('flavor', [self.flavors[0].id, 'unexist_flavor'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.compute_sdk_client.find_flavor.side_effect = [self.flavors[0], sdk_exceptions.ResourceNotFound]
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 flavors failed to delete.', str(e))
    find_calls = [mock.call(self.flavors[0].id, ignore_missing=False), mock.call('unexist_flavor', ignore_missing=False)]
    delete_calls = [mock.call(self.flavors[0].id)]
    self.compute_sdk_client.find_flavor.assert_has_calls(find_calls)
    self.compute_sdk_client.delete_flavor.assert_has_calls(delete_calls)