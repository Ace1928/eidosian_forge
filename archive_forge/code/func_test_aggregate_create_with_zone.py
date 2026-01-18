from unittest import mock
from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import aggregate
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_aggregate_create_with_zone(self):
    arglist = ['--zone', 'zone1', 'ag1']
    verifylist = [('zone', 'zone1'), ('name', 'ag1')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.create_aggregate.assert_called_once_with(name=parsed_args.name, availability_zone=parsed_args.zone)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)