from unittest import mock
from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import aggregate
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_aggregate_unset_multiple_properties(self):
    arglist = ['--property', 'unset_key1', '--property', 'unset_key2', 'ag1']
    verifylist = [('properties', ['unset_key1', 'unset_key2']), ('aggregate', 'ag1')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.set_aggregate_metadata.assert_called_once_with(self.fake_ag.id, {'unset_key1': None, 'unset_key2': None})
    self.assertIsNone(result)