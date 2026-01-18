import json
from unittest import mock
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import hypervisor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
def test_hypervisor_list_with_marker_pre_v233(self, sm_mock):
    arglist = ['--marker', 'test_hyp']
    verifylist = [('marker', 'test_hyp')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-compute-api-version 2.33 or greater is required', str(ex))