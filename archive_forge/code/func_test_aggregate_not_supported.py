from unittest import mock
from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import aggregate
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
def test_aggregate_not_supported(self, sm_mock):
    arglist = ['ag1', 'im1']
    verifylist = [('aggregate', 'ag1'), ('image', ['im1'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)