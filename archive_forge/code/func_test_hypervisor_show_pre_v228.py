import json
from unittest import mock
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import hypervisor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
def test_hypervisor_show_pre_v228(self, sm_mock):
    self.hypervisor.cpu_info = json.dumps(self.hypervisor.cpu_info)
    self.compute_sdk_client.find_hypervisor.return_value = self.hypervisor
    arglist = [self.hypervisor.name]
    verifylist = [('hypervisor', self.hypervisor.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)