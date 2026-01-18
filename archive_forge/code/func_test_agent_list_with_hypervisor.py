from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.compute.v2 import agent
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_agent_list_with_hypervisor(self):
    arglist = ['--hypervisor', 'hypervisor']
    verifylist = [('hypervisor', 'hypervisor')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.list_columns, columns)
    self.assertEqual(self.list_data, list(data))