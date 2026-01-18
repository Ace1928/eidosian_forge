from unittest import mock
from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import aggregate
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_aggregate_list_with_long(self):
    arglist = ['--long']
    vertifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, vertifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.list_columns_long, columns)
    self.assertCountEqual(self.list_data_long, tuple(data))