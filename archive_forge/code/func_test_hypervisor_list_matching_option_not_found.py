import json
from unittest import mock
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import hypervisor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
def test_hypervisor_list_matching_option_not_found(self):
    arglist = ['--matching', 'xxx']
    verifylist = [('matching', 'xxx')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.compute_sdk_client.hypervisors.side_effect = exceptions.NotFound(None)
    self.assertRaises(exceptions.NotFound, self.cmd.take_action, parsed_args)