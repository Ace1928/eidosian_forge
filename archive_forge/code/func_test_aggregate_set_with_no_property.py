from unittest import mock
from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import aggregate
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_aggregate_set_with_no_property(self):
    arglist = ['--no-property', 'ag1']
    verifylist = [('no_property', True), ('aggregate', 'ag1')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.find_aggregate.assert_called_once_with(parsed_args.aggregate, ignore_missing=False)
    self.assertNotCalled(self.compute_sdk_client.update_aggregate)
    self.compute_sdk_client.set_aggregate_metadata.assert_called_once_with(self.fake_ag.id, {'key1': None})
    self.assertIsNone(result)