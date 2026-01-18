from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_all_port_range_options(self):
    arglist = ['--dst-port', '80:80', '--icmp-type', '3', '--icmp-code', '1', self._security_group.id]
    verifylist = [('dst_port', (80, 80)), ('icmp_type', 3), ('icmp_code', 1), ('group', self._security_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)