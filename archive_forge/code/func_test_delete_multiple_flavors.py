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
def test_delete_multiple_flavors(self):
    arglist = []
    for f in self.flavors:
        arglist.append(f.id)
    verifylist = [('flavor', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.compute_sdk_client.find_flavor.side_effect = self.flavors
    result = self.cmd.take_action(parsed_args)
    find_calls = [mock.call(i.id, ignore_missing=False) for i in self.flavors]
    delete_calls = [mock.call(i.id) for i in self.flavors]
    self.compute_sdk_client.find_flavor.assert_has_calls(find_calls)
    self.compute_sdk_client.delete_flavor.assert_has_calls(delete_calls)
    self.assertIsNone(result)