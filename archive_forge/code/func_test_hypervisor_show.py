import json
from unittest import mock
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import hypervisor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
def test_hypervisor_show(self, sm_mock):
    arglist = [self.hypervisor.name]
    verifylist = [('hypervisor', self.hypervisor.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns_v288, columns)
    self.assertCountEqual(self.data_v288, data)